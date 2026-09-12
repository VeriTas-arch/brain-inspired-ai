"""Train a single agent on a single Atari game."""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from biai.atari.algorithms import (
    DEFAULT_DQN_LEARNING_STARTS,
    DQNAgent,
    PPOAgent,
)
from biai.atari.environments import AtariEnv, make_vector_atari_env, observation_protocol
from biai.atari.training import (
    DEFAULT_MAX_EPISODE_STEPS,
    DQNCollector,
    DQNMetrics,
    MetricsPlotter,
    PPOCollector,
    PPOLearner,
    ReplayBuffer,
    VideoRecorder,
    configure_ppo_runtime,
    dqn_updates_due,
    evaluation_schedule,
    run_evaluation_episodes,
    seed_everything,
)
from biai.atari.training.results import case_name, file_digest, prepare_case, result_directory
from biai.paths import RESULTS_DIR


def train_single_game(
    game_name: str = "Pong-v5",
    algorithm: str = "dqn",
    num_steps: int = 500000,
    batch_size: int = 32,
    num_envs: int = 1,
    env_backend: str = "sync",
    env_threads: int = 4,
    compile_ppo: bool = False,
    compile_dqn: bool = False,
    save_video: bool = False,
    seed: int = 0,
    deterministic: bool = False,
    eval_interval: int = 0,
    eval_points: int = 0,
    eval_episodes: int = 10,
    output_dir: Path | None = None,
):
    """Train agent on a single game."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if env_backend not in {"sync", "async", "ale"}:
        raise ValueError("env_backend must be sync, async, or ale")
    if algorithm != "ppo" and compile_ppo:
        raise ValueError("compile_ppo requires PPO")
    if num_steps <= 0 or num_steps % num_envs != 0:
        raise ValueError("Positive steps must be divisible by num_envs")
    if eval_interval < 0 or eval_episodes <= 0:
        raise ValueError("eval_interval must be nonnegative and eval_episodes positive")
    if compile_dqn and algorithm != "dqn":
        raise ValueError("compile_dqn requires DQN")
    evaluation_steps = evaluation_schedule(
        num_steps,
        points=eval_points,
        interval=eval_interval,
        step_size=num_envs * (128 if algorithm == "ppo" else 1),
    )
    exp_dir = prepare_case(
        output_dir,
        protocol="single",
        algorithm=algorithm,
        games=[game_name],
        seed=seed,
        num_steps=num_steps,
        batch_size=batch_size,
        num_envs=num_envs,
        env_backend=env_backend,
        env_threads=env_threads,
        compile_ppo=compile_ppo,
        compile_dqn=compile_dqn,
        save_video=save_video,
        deterministic=deterministic,
        eval_interval=eval_interval,
        eval_points=eval_points,
        eval_episodes=eval_episodes,
        eval_max_steps=DEFAULT_MAX_EPISODE_STEPS,
        environment_protocol=observation_protocol(env_backend),
    )
    started_at = time.perf_counter()
    configure_ppo_runtime(env_backend)
    seed_everything(seed, deterministic=deterministic)
    print(f"Training {algorithm.upper()} on {game_name}")

    env = (
        make_vector_atari_env(
            game_name,
            num_envs,
            backend=env_backend,
            num_threads=env_threads,
            render_mode=None,
            seed=seed,
        )
        if algorithm == "ppo" or num_envs > 1 or env_backend != "sync"
        else AtariEnv(game_name, render_mode=None, seed=seed)
    )

    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_fraction=0.10,
            total_timesteps=num_steps,
            target_update_freq=1000,
            tau=1.0,
        )
        agent.configure_runtime(compile_enabled=compile_dqn)
        learning_starts = DEFAULT_DQN_LEARNING_STARTS
        train_frequency = 4
        buffer = ReplayBuffer(capacity=100000, seed=(seed, 0, 1))
    else:
        agent = PPOAgent(
            state_dim=4,
            action_dim=env.action_space,
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

    video_recorder = None
    if save_video:
        video_path = exp_dir / "videos" / "training.mp4"
        video_recorder = VideoRecorder(
            str(video_path),
            fps=30,
        )

    metrics_plotter = MetricsPlotter()
    episode_rewards = []

    if algorithm == "ppo":
        agent.configure_runtime(compile_enabled=compile_ppo)
        learner = PPOLearner(
            agent,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            compile_policy=compile_ppo,
        )

        def record_ppo_frame(transition_count: int, frame: torch.Tensor) -> None:
            if video_recorder is not None and (transition_count // num_envs) % 2 == 0:
                video_recorder.add_frame(frame.cpu().numpy())

        collector = PPOCollector(
            env,
            learner,
            rollout_length=rollout_length,
            frame_callback=record_ppo_frame if video_recorder is not None else None,
        )
        print(f"PPO runtime: env_backend={env_backend}, compiled={compile_ppo}")
    else:
        collector = DQNCollector(
            env, agent, buffer, vectorized=num_envs > 1 or env_backend != "sync"
        )
    episode_count = 0

    pbar = tqdm(total=num_steps, desc="Training")

    def record_dqn_metrics(metrics):
        pbar.set_postfix(metrics, refresh=False)
        for name in ("loss", "epsilon", "q_value"):
            if name in metrics:
                metrics_plotter.add_metric(name, metrics[name])

    dqn_metrics = DQNMetrics(agent, record_dqn_metrics) if algorithm == "dqn" else None
    step = 0
    evaluations = []
    evaluation_targets = iter(evaluation_steps)
    next_evaluation = next(evaluation_targets, None)

    try:
        while step < num_steps:
            if algorithm == "dqn":
                agent.global_step = step
                completed, frame = collector.collect()
                if video_recorder is not None and (step // num_envs) % 2 == 0:
                    video_recorder.add_frame(frame.cpu().numpy())
                for _ in range(
                    dqn_updates_due(
                        step, num_envs, learning_starts=learning_starts, frequency=train_frequency
                    )
                ):
                    if buffer.is_ready(batch_size):
                        agent.update(buffer.sample(batch_size), metrics_sink=dqn_metrics.record)
                step += num_envs
                pbar.update(num_envs)
                episode_rewards.extend(completed)
                for reward in completed:
                    metrics_plotter.add_metric("episode_reward", reward)

            else:
                rollout = collector.collect(num_steps - step)
                metrics = learner.update(rollout)
                step += rollout.transition_count
                episode_count = collector.episode_count
                episode_rewards.extend(rollout.episode_returns)
                for reward_value in rollout.episode_returns:
                    metrics_plotter.add_metric("episode_reward", reward_value)

                pbar.set_postfix({**metrics, "episodes": episode_count}, refresh=False)
                for metric_name in ("policy_loss", "value_loss", "entropy"):
                    if metric_name in metrics:
                        metrics_plotter.add_metric(metric_name, metrics[metric_name])
                pbar.update(rollout.transition_count)
            if next_evaluation is not None and step >= next_evaluation:
                if dqn_metrics is not None:
                    dqn_metrics.flush()
                evaluation_env = AtariEnv(game_name, seed=seed, training=False, backend=env_backend)
                try:
                    rewards = run_evaluation_episodes(
                        agent, evaluation_env, eval_episodes, DEFAULT_MAX_EPISODE_STEPS
                    )
                finally:
                    evaluation_env.close()
                evaluations.append(
                    {
                        "step": step,
                        "elapsed_seconds": time.perf_counter() - started_at,
                        "rewards": rewards,
                        "mean_raw_reward": sum(rewards) / len(rewards),
                    }
                )
                (exp_dir / "learning.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "deterministic": deterministic,
                            "eval_points": eval_points,
                            "evaluation_steps": evaluation_steps,
                            "evaluations": evaluations,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[Eval] step={step}, mean raw reward={evaluations[-1]['mean_raw_reward']}")
                next_evaluation = next(evaluation_targets, None)
    finally:
        if dqn_metrics is not None:
            dqn_metrics.flush()
        pbar.close()
        env.close()
        if video_recorder is not None:
            video_recorder.close()

    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        last_n = min(20, len(episode_rewards))
        avg_last_n = sum(episode_rewards[-last_n:]) / last_n
        print(f"\n{'=' * 60}")
        print("Training Summary")
        print(f"{'=' * 60}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average episode reward: {avg_reward:.2f}")
        print(f"Average reward over last {last_n} episodes: {avg_last_n:.2f}")

        if len(episode_rewards) >= 20:
            first_10 = sum(episode_rewards[:10]) / 10
            last_10 = sum(episode_rewards[-10:]) / 10
            print("\nReward Progression:")
            print(f"  First 10 episodes: {first_10:.2f}")
            print(f"  Last 10 episodes: {last_10:.2f}")
            improvement = last_10 - first_10
            print(f"  Improvement: {improvement:+.2f}")
            if improvement > 0:
                print("  ✓ Agent is learning!")
            elif improvement < -1:
                print("  ⚠ Agent performance decreased")
            else:
                print("  → Agent performance stable")

        best_ep = max(episode_rewards)
        worst_ep = min(episode_rewards)
        print(f"\nBest episode reward: {best_ep:.2f}")
        print(f"Worst episode reward: {worst_ep:.2f}")
        print(f"{'=' * 60}")

    metrics_output_path = exp_dir / "figures" / "metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Metrics plot saved to {metrics_output_path}")

    checkpoint_path = exp_dir / "checkpoints" / "final.pt"
    agent.save(str(checkpoint_path))
    print(f"Agent saved to {checkpoint_path}")

    (exp_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": file_digest(checkpoint_path),
                "algorithm": algorithm,
                "seed": seed,
                "deterministic": deterministic,
                "total_steps": step,
                "eval_points": eval_points,
                "evaluation_steps": evaluation_steps,
                "num_envs": num_envs,
                "env_backend": env_backend,
                "env_threads": env_threads,
                "environment_protocol": agent.environment_protocol,
                "compile_ppo": compile_ppo,
                "compile_dqn": compile_dqn,
                "replay_sampler": "independent_pcg64_without_replacement"
                if algorithm == "dqn"
                else None,
                "replay_seed": [seed, 0, 1] if algorithm == "dqn" else None,
                "optimizer_steps": agent.update_count if algorithm == "dqn" else None,
                "elapsed_seconds": time.perf_counter() - started_at,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pong-v5", help="Game name")
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-backend", choices=("sync", "async", "ale"), default="sync")
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--compile-ppo", action="store_true")
    parser.add_argument("--compile-dqn", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument(
        "--eval-points",
        type=int,
        default=0,
        help="Evaluation points per task budget; exclusive with --eval-interval",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--save-video", action="store_true", help="Enable video recording (disabled by default)"
    )

    parser.add_argument("--output-dir", type=Path, help="Case directory; default: results/<case>")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing results after success"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_DIR / case_name(
        "single", args.algorithm, game=args.game, seed=args.seed
    )
    with result_directory(output_dir, force=args.force) as staging:
        train_single_game(
            output_dir=staging,
            game_name=args.game,
            algorithm=args.algorithm,
            num_steps=args.steps,
            batch_size=args.batch_size,
            num_envs=args.num_envs,
            env_backend=args.env_backend,
            env_threads=args.env_threads,
            compile_ppo=args.compile_ppo,
            compile_dqn=args.compile_dqn,
            save_video=args.save_video,
            seed=args.seed,
            deterministic=args.deterministic,
            eval_interval=args.eval_interval,
            eval_points=args.eval_points,
            eval_episodes=args.eval_episodes,
        )
    print(f"Results saved to {output_dir}")
