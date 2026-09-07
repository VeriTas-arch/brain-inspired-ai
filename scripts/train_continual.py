"""Continual learning on multiple Atari games."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from algorithms import (
    DEFAULT_DQN_LEARNING_STARTS,
    EWCWrapper,
    MultiHeadDQNAgent,
    MultiHeadPPOAgent,
)
from environments import AtariEnv, make_vector_atari_env
from training import (
    DEFAULT_MAX_EPISODE_STEPS,
    PPOCollector,
    PPOLearner,
    configure_ppo_runtime,
    flatten_rollout_data,
    run_evaluation_episodes,
)
from utils import MetricsPlotter, ReplayBuffer, VideoRecorder, seed_everything


def build_evaluation_report(eval_history: dict, games: list[str]) -> dict:
    """Build a stage-by-task score matrix without aggregating across games."""
    history_by_game = {
        game: {stage: float(reward) for stage, reward in eval_history.get(game, [])}
        for game in games
    }
    score_matrix = [
        {
            "stage": stage,
            "after_task": trained_game,
            "scores": {game: history_by_game[game].get(stage) for game in games},
        }
        for stage, trained_game in enumerate(games, start=1)
    ]

    per_task = {}
    for game in games:
        history = sorted(eval_history.get(game, []), key=lambda item: item[0])
        if not history:
            continue
        scores = [float(reward) for _, reward in history]
        per_task[game] = {
            "score_after_learning": scores[0],
            "best_score": max(scores),
            "final_score": scores[-1],
            "forgetting": max(scores) - scores[-1],
        }

    return {
        "score_units": "raw_environment_reward",
        "score_matrix": score_matrix,
        "per_task": per_task,
    }


def plot_forgetting_curves(eval_history: dict, output_path: Path) -> bool:
    """Plot forgetting curves using evaluation history."""
    has_data = False
    max_stage = 0

    plt.figure(figsize=(8, 5))
    for game_name, history in eval_history.items():
        if not history:
            continue
        history = sorted(history, key=lambda item: item[0])
        stages, rewards = zip(*history)
        plt.plot(stages, rewards, marker="o", label=game_name)
        max_stage = max(max_stage, stages[-1])
        has_data = True

    if not has_data:
        print("No evaluation history collected; skipping forgetting curve plot.")
        plt.close()
        return False

    plt.xlabel("Training Stage")
    plt.ylabel("Average Reward")
    plt.title("Continual Evaluation (Forgetting Curves)")
    plt.xticks(range(1, max_stage + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Continual evaluation (forgetting) plot saved: {output_path}")
    return True


def train_continual(
    games: list = None,
    algorithm: str = "dqn",
    use_ewc: bool = False,
    ewc_lambda: float = 0.4,
    steps_per_game: int = 50000,
    batch_size: int = 32,
    num_envs: int = 1,
    env_backend: str = "sync",
    compile_ppo: bool = False,
    eval_episodes: int = 5,
    eval_max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    save_video: bool = False,
    seed: int = 0,
):
    """Train agent on multiple games sequentially."""
    if games is None:
        games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if algorithm != "ppo" and num_envs != 1:
        raise ValueError("num_envs greater than one is currently supported only for PPO")
    if env_backend not in {"sync", "async"}:
        raise ValueError("env_backend must be 'sync' or 'async'")
    if algorithm != "ppo" and (env_backend != "sync" or compile_ppo):
        raise ValueError("env_backend and compile_ppo options are supported only for PPO")
    if algorithm == "ppo" and steps_per_game % num_envs != 0:
        raise ValueError("PPO steps_per_game must be divisible by num_envs")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if eval_max_steps <= 0:
        raise ValueError("eval_max_steps must be positive")
    configure_ppo_runtime(env_backend)
    seed_everything(seed)
    game_seeds = {game: seed + index for index, game in enumerate(games)}
    run_name = f"{algorithm}_ewc{use_ewc}"
    exp_root = Path("outputs") / "continual" / run_name / f"seed-{seed}"

    print(f"Continual Learning: {algorithm.upper()} on {games}")
    print(f"EWC: {use_ewc}")
    if use_ewc:
        print(f"EWC Lambda: {ewc_lambda}")

    continual_metrics = {}
    eval_history = {game: [] for game in games}
    ewc_diagnostics = {}
    agent = None
    for game_idx, game_name in enumerate(games):
        print(f"\n=== Task {game_idx + 1}/{len(games)}: {game_name} ===")

        env = (
            make_vector_atari_env(
                game_name,
                num_envs,
                backend=env_backend,
                render_mode=None,
                seed=game_seeds[game_name],
            )
            if algorithm == "ppo"
            else AtariEnv(game_name, render_mode=None, seed=game_seeds[game_name])
        )
        action_dim = env.action_space
        task_id = game_name

        if agent is None:
            if algorithm == "dqn":
                base_agent = MultiHeadDQNAgent(
                    state_dim=4,
                    lr=1e-4,
                    gamma=0.99,
                )
                base_agent.register_task(task_id, action_dim)
                base_agent.set_task(task_id)
            else:  # ppo
                base_agent = MultiHeadPPOAgent(
                    state_dim=4,
                    lr=2.5e-4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_coef=0.1,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                )
                base_agent.register_task(task_id, action_dim)
                base_agent.set_task(task_id)

            agent = EWCWrapper(base_agent, ewc_lambda=ewc_lambda) if use_ewc else base_agent
        else:
            if isinstance(agent, EWCWrapper):
                if isinstance(agent.agent, (MultiHeadDQNAgent, MultiHeadPPOAgent)):
                    agent.register_task(task_id, action_dim)
                    agent.set_task(task_id)
            elif isinstance(agent, (MultiHeadDQNAgent, MultiHeadPPOAgent)):
                agent.register_task(task_id, action_dim)
                agent.set_task(task_id)

        # Initialize buffers and training parameters based on algorithm
        if algorithm == "dqn":
            learning_starts = DEFAULT_DQN_LEARNING_STARTS
            train_frequency = 4
            buffer = ReplayBuffer(capacity=100000)
        else:  # ppo
            learning_starts = 0
            train_frequency = 1
            rollout_length = 128
            update_epochs = 4
            minibatch_size = batch_size

        exp_dir = exp_root / game_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        video_recorder = None
        if save_video:
            video_recorder = VideoRecorder(str(exp_dir / "training.mp4"), fps=30)

        if algorithm == "ppo":
            learner = PPOLearner(
                agent,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                compile_policy=compile_ppo,
            )

            def record_ppo_frame(transition_count: int, frame: torch.Tensor) -> None:
                if video_recorder is not None and transition_count % 2 == 0:
                    video_recorder.add_frame(frame.cpu().numpy())

            collector = PPOCollector(
                env,
                learner,
                rollout_length=rollout_length,
                frame_callback=record_ppo_frame if video_recorder is not None else None,
            )
        else:
            state = env.reset()
            episode_reward = 0.0
        episode_rewards = []
        episode_count = 0
        step = 0
        last_rollout_data = None

        pbar = tqdm(total=steps_per_game, desc=f"Training on {game_name}")

        while step < steps_per_game:
            if algorithm == "dqn":
                action = agent.select_action(state)

                next_state, reward, terminated, truncated = env.step(action)
                episode_done = terminated or truncated
                episode_reward += reward
                buffer.add(state, action, reward, next_state, terminated)

                if video_recorder is not None and step % 2 == 0:
                    frame = state[0].cpu().numpy() if isinstance(state, torch.Tensor) else state[0]
                    video_recorder.add_frame(frame)

                if step >= learning_starts and step % train_frequency == 0:
                    if buffer.is_ready(batch_size):
                        batch = buffer.sample(batch_size)
                        metrics = agent.update(batch)
                        pbar.set_postfix(metrics)

                        if "loss" in metrics:
                            continual_metrics.setdefault(game_name + "_loss", []).append(
                                metrics["loss"]
                            )
                        if "epsilon" in metrics:
                            continual_metrics.setdefault(game_name + "_epsilon", []).append(
                                metrics["epsilon"]
                            )
                        if "q_value" in metrics:
                            continual_metrics.setdefault(game_name + "_q_value", []).append(
                                metrics["q_value"]
                            )
                        if "ewc_loss" in metrics:
                            continual_metrics.setdefault(game_name + "_ewc_loss", []).append(
                                metrics["ewc_loss"]
                            )

                state = next_state
                step += 1
                pbar.update(1)

                if episode_done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    state = env.reset()

            else:  # ppo
                rollout = collector.collect(steps_per_game - step)
                metrics = learner.update(rollout)
                step += rollout.transition_count
                episode_count = collector.episode_count
                episode_rewards.extend(rollout.episode_returns)
                if step == steps_per_game:
                    last_rollout_data = flatten_rollout_data(rollout.data, clone=True)

                pbar.set_postfix({**metrics, "episodes": episode_count})
                for metric_name in ("policy_loss", "value_loss", "entropy", "ewc_loss"):
                    if metric_name in metrics:
                        continual_metrics.setdefault(game_name + f"_{metric_name}", []).append(
                            metrics[metric_name]
                        )
                pbar.update(rollout.transition_count)

        pbar.close()

        if episode_rewards:
            continual_metrics.setdefault(game_name + "_episode_reward", []).extend(episode_rewards)

        for eval_game in games[: game_idx + 1]:
            eval_env = AtariEnv(
                eval_game,
                render_mode=None,
                training=False,
                seed=game_seeds[eval_game],
            )
            eval_task_id = eval_game
            if isinstance(agent, EWCWrapper) and hasattr(agent.agent, "set_task"):
                agent.set_task(eval_task_id)
            elif hasattr(agent, "set_task"):
                agent.set_task(eval_task_id)

            episode_eval_rewards = run_evaluation_episodes(
                agent,
                eval_env,
                eval_episodes,
                eval_max_steps,
            )
            avg_eval_reward = sum(episode_eval_rewards) / len(episode_eval_rewards)
            eval_env.close()

            eval_history[eval_game].append((game_idx + 1, avg_eval_reward))
            print(
                f"[Eval] After task {game_name}, on {eval_game}: avg reward {avg_eval_reward:.2f}"
            )

        if video_recorder is not None:
            video_recorder.save(format="mp4")
            print(f"Video saved: {exp_dir / 'training.mp4'}")

        if use_ewc:
            sample_batch = None
            if algorithm == "dqn" and buffer.is_ready(batch_size):
                fisher_rng = np.random.default_rng([seed, game_idx])
                sample_batch = buffer.sample(min(batch_size, 100), rng=fisher_rng)
            elif algorithm == "ppo":
                sample_batch = last_rollout_data

            ewc_diagnostics[game_name] = agent.consolidate_weights(sample_batch)
            print(f"Weights consolidated for EWC: {ewc_diagnostics[game_name]}")

        env.close()

    metrics_plotter = MetricsPlotter()
    for name, values in continual_metrics.items():
        for v in values:
            metrics_plotter.add_metric(name, v)

    exp_root.mkdir(parents=True, exist_ok=True)

    metrics_output_path = exp_root / "training_metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Continual metrics plot saved: {metrics_output_path}")

    eval_metrics_path = exp_root / "forgetting_eval.png"
    plot_forgetting_curves(eval_history, eval_metrics_path)

    evaluation_report = {
        "algorithm": algorithm,
        "games": games,
        "use_ewc": use_ewc,
        "ewc_lambda": ewc_lambda if use_ewc else None,
        "steps_per_game": steps_per_game,
        "batch_size": batch_size,
        "num_envs": num_envs,
        "env_backend": env_backend if algorithm == "ppo" else None,
        "compile_ppo": compile_ppo if algorithm == "ppo" else None,
        "eval_episodes": eval_episodes,
        "eval_max_steps": eval_max_steps,
        "seed": seed,
        "task_seeds": game_seeds,
        "ewc_diagnostics": ewc_diagnostics if use_ewc else None,
        **build_evaluation_report(eval_history, games),
    }
    evaluation_path = exp_root / "continual_evaluation.json"
    with evaluation_path.open("w", encoding="utf-8") as output_file:
        json.dump(evaluation_report, output_file, indent=2)
    print(f"Continual evaluation data saved: {evaluation_path}")

    # Save final agent checkpoint in a structured location
    ckpt_root = Path("checkpoints") / "continual"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / run_name / f"seed-{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))
    print(f"\nAgent saved: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games", nargs="+", default=["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    )
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--use-ewc", action="store_true")
    parser.add_argument("--ewc-lambda", type=float, default=0.4, help="EWC regularization strength")
    parser.add_argument("--steps-per-game", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-backend", choices=("sync", "async"), default="sync")
    parser.add_argument("--compile-ppo", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-video", action="store_true", help="Enable video recording (disabled by default)"
    )

    args = parser.parse_args()

    train_continual(
        games=args.games,
        algorithm=args.algorithm,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        steps_per_game=args.steps_per_game,
        batch_size=args.batch_size,
        num_envs=args.num_envs,
        env_backend=args.env_backend,
        compile_ppo=args.compile_ppo,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        save_video=args.save_video,
        seed=args.seed,
    )
