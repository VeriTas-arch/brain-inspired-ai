"""Multi-task joint training on multiple Atari games."""

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tqdm import tqdm

from algorithms import (
    DEFAULT_DQN_LEARNING_STARTS,
    MultiHeadDQNAgent,
    MultiHeadPPOAgent,
)
from environments import AtariEnv, make_vector_atari_env, observation_protocol
from training import (
    DQNCollector,
    DQNMetrics,
    MetricsPlotter,
    PPOCollector,
    PPOLearner,
    ReplayBuffer,
    VideoRecorder,
    configure_ppo_runtime,
    dqn_updates_due,
    seed_everything,
)
from training.results import case_name, file_digest, prepare_case, result_directory


def train_multitask(
    games: list = None,
    algorithm: str = "dqn",
    total_steps: int = 150000,
    batch_size: int = 32,
    save_video: bool = False,
    seed: int = 0,
    deterministic: bool = False,
    compile_ppo: bool = False,
    compile_dqn: bool = False,
    num_envs: int = 1,
    env_backend: str = "sync",
    env_threads: int = 4,
    output_dir: Path | None = None,
):
    """Train agent on multiple games jointly (random task sampling per iteration)."""
    if compile_ppo and algorithm != "ppo":
        raise ValueError("compile_ppo requires PPO")
    if games is None:
        games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if compile_dqn and algorithm != "dqn":
        raise ValueError("compile_dqn requires DQN")
    if not games or len(set(games)) != len(games):
        raise ValueError("games must be nonempty and distinct")
    if num_envs <= 0 or total_steps <= 0 or total_steps % num_envs:
        raise ValueError("Positive total_steps must be divisible by positive num_envs")
    if env_backend not in {"sync", "async", "ale"} or env_threads <= 0:
        raise ValueError("Require a supported environment backend and positive env_threads")
    exp_dir = prepare_case(
        output_dir,
        protocol="multitask",
        algorithm=algorithm,
        games=games,
        seed=seed,
        total_steps=total_steps,
        batch_size=batch_size,
        num_envs=num_envs,
        env_backend=env_backend,
        env_threads=env_threads,
        compile_ppo=compile_ppo,
        compile_dqn=compile_dqn,
        save_video=save_video,
        deterministic=deterministic,
        environment_protocol=observation_protocol(env_backend),
    )
    started_at = time.perf_counter()
    configure_ppo_runtime(env_backend)
    seed_everything(seed, deterministic=deterministic)

    print(f"Multi-task Joint Training: {algorithm.upper()} on {games}")
    print(f"Total steps: {total_steps}")

    # Initialize environments and get action dimensions
    envs = {}
    action_dims = {}
    for game_index, game_name in enumerate(games):
        env = (
            make_vector_atari_env(
                game_name,
                num_envs,
                backend=env_backend,
                seed=seed + game_index,
                num_threads=env_threads,
            )
            if algorithm == "ppo" or num_envs > 1 or env_backend != "sync"
            else AtariEnv(game_name, seed=seed + game_index)
        )
        envs[game_name] = env
        action_dims[game_name] = env.action_space

    # Initialize agent with multi-head architecture
    if algorithm == "dqn":
        agent = MultiHeadDQNAgent(
            state_dim=4,
            lr=1e-4,
            gamma=0.99,
        )
        learning_starts = DEFAULT_DQN_LEARNING_STARTS
        train_frequency = 4
        buffers = {
            game: ReplayBuffer(capacity=100000, seed=(seed, index, 1))
            for index, game in enumerate(games)
        }
    else:  # ppo
        agent = MultiHeadPPOAgent(
            state_dim=4,
            lr=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        learning_starts = 0
        train_frequency = 1
        rollout_length = 128
        update_epochs = 4
        minibatch_size = batch_size

    agent.environment_protocol = observation_protocol(env_backend)
    # Register all tasks
    for game_name in games:
        agent.register_task(game_name, action_dims[game_name])

    if algorithm == "dqn":
        agent.configure_runtime(compile_enabled=compile_dqn)

    if algorithm == "ppo":
        agent.configure_runtime(compile_enabled=compile_ppo)
        learner = PPOLearner(
            agent,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            compile_policy=compile_ppo,
        )

    # Per-game video recorders (optional)
    video_recorders = {}
    if save_video:
        for game_name in games:
            game_dir = exp_dir / "videos" / game_name
            game_dir.mkdir(parents=True, exist_ok=True)
            video_recorders[game_name] = VideoRecorder(str(game_dir / "training.mp4"), fps=30)

    metrics_plotter = MetricsPlotter()
    episode_rewards = defaultdict(list)

    collectors = {
        game: (
            PPOCollector(envs[game], learner, rollout_length=rollout_length)
            if algorithm == "ppo"
            else DQNCollector(
                envs[game], agent, buffers[game], vectorized=num_envs > 1 or env_backend != "sync"
            )
        )
        for game in games
    }
    if save_video and algorithm == "ppo":
        for game, collector in collectors.items():
            recorder = video_recorders[game]
            collector.frame_callback = lambda count, frame, recorder=recorder: (
                recorder.add_frame(frame.cpu().numpy()) if (count // num_envs) % 2 == 0 else None
            )
    pbar = tqdm(total=total_steps, desc="Multi-task Training")

    def record_dqn_metrics(game, metrics):
        for name in ("loss", "epsilon", "q_value"):
            if name in metrics:
                metrics_plotter.add_metric(f"{game}_{name}", metrics[name])

    dqn_metrics = (
        {
            game: DQNMetrics(agent, lambda metrics, game=game: record_dqn_metrics(game, metrics))
            for game in games
        }
        if algorithm == "dqn"
        else {}
    )
    step = 0
    task_steps = {game: 0 for game in games}
    try:
        while step < total_steps:
            # One game supplies the entire batch. PPO finishes collection before any update.
            current_game = str(np.random.choice(games))
            agent.set_task(current_game)
            collector = collectors[current_game]
            if algorithm == "dqn":
                completed, frame = collector.collect()
                metrics = {}
                for _ in range(
                    dqn_updates_due(
                        step,
                        num_envs,
                        learning_starts=learning_starts,
                        frequency=train_frequency,
                        post_increment=True,
                    )
                ):
                    buffer = buffers[current_game]
                    if buffer.is_ready(batch_size):
                        agent.update(
                            buffer.sample(batch_size), metrics_sink=dqn_metrics[current_game].record
                        )
                transitions = num_envs
                if save_video and (task_steps[current_game] // num_envs) % 2 == 0:
                    video_recorders[current_game].add_frame(frame.cpu().numpy())
            else:
                rollout = collector.collect(total_steps - step)
                metrics = learner.update(rollout)
                completed, transitions = rollout.episode_returns, rollout.transition_count
                for name in ("policy_loss", "value_loss", "entropy"):
                    metrics_plotter.add_metric(f"{current_game}_{name}", metrics[name])
            step += transitions
            task_steps[current_game] += transitions
            episode_rewards[current_game].extend(completed)
            for reward in completed:
                metrics_plotter.add_metric(f"{current_game}_episode_reward", reward)
            pbar.update(transitions)
            pbar.set_postfix(
                {**metrics, "game": current_game[:8], "episodes": collector.episode_count},
                refresh=False,
            )
    finally:
        for pending in dqn_metrics.values():
            pending.flush()
        pbar.close()
        with ExitStack() as resources:
            for recorder in video_recorders.values():
                resources.callback(recorder.close)
            for env in envs.values():
                resources.callback(env.close)

    # Print training summary
    print(f"\n{'=' * 60}")
    print("Multi-task Training Summary")
    print(f"{'=' * 60}")
    for game_name in games:
        if episode_rewards[game_name]:
            avg_reward = sum(episode_rewards[game_name]) / len(episode_rewards[game_name])
            last_n = min(20, len(episode_rewards[game_name]))
            avg_last_n = sum(episode_rewards[game_name][-last_n:]) / last_n if last_n > 0 else 0.0
            print(f"\n{game_name}:")
            print(f"  Total episodes: {len(episode_rewards[game_name])}")
            print(f"  Average episode reward: {avg_reward:.2f}")
            print(f"  Average reward over last {last_n} episodes: {avg_last_n:.2f}")

    # Save metrics plot
    metrics_output_path = exp_dir / "figures" / "training_metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"\nMetrics plot saved: {metrics_output_path}")

    checkpoint_path = exp_dir / "checkpoints" / "final.pt"
    agent.save(str(checkpoint_path))
    print(f"Agent saved: {checkpoint_path}")

    (exp_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": file_digest(checkpoint_path),
                "algorithm": algorithm,
                "seed": seed,
                "deterministic": deterministic,
                "total_steps": step,
                "elapsed_seconds": time.perf_counter() - started_at,
                "task_steps": task_steps,
                "num_envs": num_envs,
                "env_backend": env_backend,
                "env_threads": env_threads,
                "environment_protocol": agent.environment_protocol,
                "task_sampling": "one_task_per_rollout"
                if algorithm == "ppo"
                else "one_task_per_environment_batch",
                "replay_sampler": "independent_pcg64_without_replacement"
                if algorithm == "dqn"
                else None,
                "replay_seeds": {game: [seed, index, 1] for index, game in enumerate(games)}
                if algorithm == "dqn"
                else None,
                "optimizer_steps": agent.update_count if algorithm == "dqn" else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save model checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games", nargs="+", default=["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    )
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument(
        "--steps", type=int, default=150000, help="Total training steps across all games"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-backend", choices=("sync", "async", "ale"), default="sync")
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile-ppo", action="store_true")
    parser.add_argument("--compile-dqn", action="store_true")
    parser.add_argument(
        "--save-video", action="store_true", help="Enable video recording (disabled by default)"
    )

    parser.add_argument("--output-dir", type=Path, help="Case directory; default: results/<case>")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing results after success"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path("results") / case_name(
        "multitask", args.algorithm, seed=args.seed
    )
    with result_directory(output_dir, force=args.force) as staging:
        train_multitask(
            output_dir=staging,
            games=args.games,
            algorithm=args.algorithm,
            total_steps=args.steps,
            batch_size=args.batch_size,
            save_video=args.save_video,
            seed=args.seed,
            deterministic=args.deterministic,
            compile_ppo=args.compile_ppo,
            compile_dqn=args.compile_dqn,
            num_envs=args.num_envs,
            env_backend=args.env_backend,
            env_threads=args.env_threads,
        )
    print(f"Results saved to {output_dir}")
