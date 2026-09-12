"""Continual learning on multiple Atari games."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from biai.atari.algorithms import (
    DEFAULT_DQN_LEARNING_STARTS,
    EWCWrapper,
    MultiHeadDQNAgent,
    MultiHeadPPOAgent,
)
from biai.atari.algorithms.subspace_projection import AdamSubspaceProjection, build_input_subspaces
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
    flatten_rollout_data,
    run_evaluation_episodes,
    seed_everything,
)
from biai.atari.training.results import (
    case_name,
    file_digest,
    parameter_digest,
    prepare_case,
    result_directory,
)
from biai.paths import RESULTS_DIR


def collect_gpm_states(
    agent, *, num_envs, env_backend, seed, collection_steps, samples, env_threads=4
):
    """Sample a frozen task policy without updates or changes to the training RNG."""
    is_dqn = isinstance(agent, MultiHeadDQNAgent)
    environment = (
        AtariEnv(agent.current_task, seed=seed, training=False, backend=env_backend)
        if is_dqn and num_envs == 1
        else make_vector_atari_env(
            agent.current_task,
            num_envs,
            backend=env_backend,
            seed=seed,
            training=False,
            num_threads=env_threads,
        )
    )
    devices = [agent.device.index or 0] if agent.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(collection_steps, generator=generator)[:samples]
            vectorized = not is_dqn or num_envs > 1
            state = environment.reset()
            shape = state.shape[1:] if vectorized else state.shape
            states = torch.empty((len(indices), *shape), dtype=state.dtype)
            selected = {step: position for position, step in enumerate(indices.tolist())}
            with torch.inference_mode():
                for step in range(0, collection_steps, num_envs):
                    for offset in range(num_envs):
                        position = selected.get(step + offset)
                        if position is not None:
                            states[position].copy_(state[offset] if vectorized else state)
                    if vectorized:
                        actions = agent.select_actions(state, deterministic=is_dqn)
                        state = environment.step_and_reset(actions.cpu()).observations
                    else:
                        action = agent.select_action(state, deterministic=True)
                        state, _, terminated, truncated = environment.step(action)
                        if terminated or truncated:
                            state = environment.reset()
            return states
    finally:
        environment.close()


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
    env_threads: int = 4,
    compile_ppo: bool = False,
    compile_dqn: bool = False,
    eval_episodes: int = 5,
    eval_max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    save_video: bool = False,
    seed: int = 0,
    deterministic: bool = False,
    eval_interval: int = 0,
    eval_points: int = 0,
    method: str | None = None,
    task_steps: list[int] | None = None,
    gpm_threshold: float = 0.995,
    gpm_samples: int = 2048,
    gpm_collection_steps: int = 32768,
    output_dir: Path | None = None,
):
    """Train every task from one fresh agent; no pretrained checkpoint is required."""
    if games is None:
        games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    if not games or len(set(games)) != len(games):
        raise ValueError("games must be a nonempty sequence of distinct tasks")
    if method is None:
        method = "ewc" if use_ewc else "finetune"
    if method not in {"finetune", "ewc", "gpm"}:
        raise ValueError("method must be 'finetune', 'ewc', or 'gpm'")
    if use_ewc and method != "ewc":
        raise ValueError("use_ewc cannot be combined with another method")
    use_ewc = method == "ewc"
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if env_backend not in {"sync", "async", "ale"}:
        raise ValueError("env_backend must be sync, async, or ale")
    if algorithm != "ppo" and compile_ppo:
        raise ValueError("compile_ppo requires PPO")
    budgets = list(task_steps) if task_steps is not None else [steps_per_game] * len(games)
    if len(budgets) != len(games) or any(steps <= 0 for steps in budgets):
        raise ValueError("task_steps must provide one positive transition budget per game")
    if any(steps % num_envs != 0 for steps in budgets):
        raise ValueError("Task budgets must be divisible by num_envs")
    if method == "gpm":
        if not 0 < gpm_threshold <= 1:
            raise ValueError("gpm_threshold must be in (0, 1]")
        if not 0 < gpm_samples <= gpm_collection_steps:
            raise ValueError("Require 0 < gpm_samples <= gpm_collection_steps")
        if gpm_collection_steps % num_envs != 0:
            raise ValueError("gpm_collection_steps must be divisible by num_envs")
    if eval_interval < 0:
        raise ValueError("eval_interval must be nonnegative")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if eval_max_steps <= 0:
        raise ValueError("eval_max_steps must be positive")
    if compile_dqn and algorithm != "dqn":
        raise ValueError("compile_dqn requires DQN")
    evaluation_steps = {
        game: evaluation_schedule(
            budget,
            points=eval_points,
            interval=eval_interval,
            step_size=num_envs * (128 if algorithm == "ppo" else 1),
        )
        for game, budget in zip(games, budgets, strict=True)
    }
    exp_root = prepare_case(
        output_dir,
        protocol="continual",
        algorithm=algorithm,
        method=method,
        games=games,
        seed=seed,
        task_steps=budgets,
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
        eval_max_steps=eval_max_steps,
        ewc_lambda=ewc_lambda if use_ewc else None,
        gpm_threshold=gpm_threshold if method == "gpm" else None,
        gpm_samples=gpm_samples if method == "gpm" else None,
        gpm_collection_steps=gpm_collection_steps if method == "gpm" else None,
        environment_protocol=observation_protocol(env_backend),
    )
    started_at = time.perf_counter()
    configure_ppo_runtime(env_backend)
    seed_everything(seed, deterministic=deterministic)
    game_seeds = {game: seed + index for index, game in enumerate(games)}

    print(f"Continual Learning: {algorithm.upper()} on {games}")
    print(f"Method: {method}; task transition budgets: {budgets}")
    if use_ewc:
        print(f"EWC Lambda: {ewc_lambda}")

    continual_metrics = {}
    eval_history = {game: [] for game in games}
    stage_episode_rewards = []
    head_seeds = {game: seed + 60000 + index for index, game in enumerate(games) if index}
    ewc_diagnostics = {}
    gpm_diagnostics = {}
    boundary_checkpoints = []
    checkpoint_dir = exp_root / "checkpoints"
    initialization = {}
    learning_history = {}
    subspaces = None
    agent = None
    for game_idx, game_name in enumerate(games):
        stage_steps = budgets[game_idx]
        print(f"\n=== Task {game_idx + 1}/{len(games)}: {game_name} ===")

        env = (
            make_vector_atari_env(
                game_name,
                num_envs,
                backend=env_backend,
                num_threads=env_threads,
                render_mode=None,
                seed=game_seeds[game_name],
            )
            if algorithm == "ppo" or num_envs > 1 or env_backend != "sync"
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

            base_agent.environment_protocol = observation_protocol(env_backend)
            agent = EWCWrapper(base_agent, ewc_lambda=ewc_lambda) if use_ewc else base_agent
        else:
            # Match new-head initialization without resetting rollout/minibatch RNG streams.
            devices = [agent.device.index or 0] if agent.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(head_seeds[game_name])
                agent.register_task(task_id, action_dim)
            agent.set_task(task_id)

        modules = {"backbone": base_agent.backbone} if game_idx == 0 else {}
        modules.update(
            {"head": base_agent.heads[task_id]}
            if algorithm == "dqn"
            else {"actor": base_agent.actors[task_id], "critic": base_agent.critics[task_id]}
        )
        initialization[game_name] = {
            name: parameter_digest(module) for name, module in modules.items()
        }
        (exp_root / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")

        # Initialize buffers and training parameters based on algorithm
        if algorithm == "dqn":
            agent.configure_runtime(compile_enabled=compile_dqn)
            learning_starts = DEFAULT_DQN_LEARNING_STARTS
            train_frequency = 4
            buffer = ReplayBuffer(capacity=100000, seed=(seed, game_idx, 1))
        else:  # ppo
            learning_starts = 0
            train_frequency = 1
            rollout_length = 128
            update_epochs = 4
            minibatch_size = batch_size

        exp_dir = exp_root / "figures" / game_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        video_recorder = None
        if save_video:
            video_recorder = VideoRecorder(
                str(exp_root / "videos" / f"{game_name}_training.mp4"), fps=30
            )

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
        else:
            collector = DQNCollector(
                env, agent, buffer, vectorized=num_envs > 1 or env_backend != "sync"
            )
        episode_rewards = []
        episode_count = 0
        step = 0
        last_rollout_data = None
        learning_evaluations = []
        evaluation_targets = iter(evaluation_steps[game_name])
        next_evaluation = next(evaluation_targets, None)

        # DQN normalizes inside its backbone; the GPM builder normalizes before forwarding.
        if method == "gpm":
            gpm_backbone = agent.backbone.network if algorithm == "dqn" else agent.backbone
        projection = (
            AdamSubspaceProjection(
                agent.optimizer,
                gpm_backbone,
                subspaces,
                compile_projection=compile_dqn if algorithm == "dqn" else compile_ppo,
            )
            if method == "gpm" and subspaces is not None
            else None
        )
        pbar = tqdm(total=stage_steps, desc=f"Training on {game_name}")

        def record_dqn_metrics(metrics):
            pbar.set_postfix(metrics, refresh=False)
            for name in ("loss", "epsilon", "q_value", "ewc_loss"):
                if name in metrics:
                    continual_metrics.setdefault(game_name + "_" + name, []).append(metrics[name])

        dqn_metrics = DQNMetrics(agent, record_dqn_metrics) if algorithm == "dqn" else None

        try:
            while step < stage_steps:
                if algorithm == "dqn":
                    completed, frame = collector.collect()
                    if video_recorder is not None and (step // num_envs) % 2 == 0:
                        video_recorder.add_frame(frame.cpu().numpy())
                    for _ in range(
                        dqn_updates_due(
                            step,
                            num_envs,
                            learning_starts=learning_starts,
                            frequency=train_frequency,
                        )
                    ):
                        if buffer.is_ready(batch_size):
                            agent.update(buffer.sample(batch_size), metrics_sink=dqn_metrics.record)
                    step += num_envs
                    pbar.update(num_envs)
                    episode_rewards.extend(completed)

                else:  # ppo
                    rollout = collector.collect(stage_steps - step)
                    metrics = learner.update(rollout)
                    step += rollout.transition_count
                    episode_count = collector.episode_count
                    episode_rewards.extend(rollout.episode_returns)
                    if step == stage_steps:
                        last_rollout_data = flatten_rollout_data(rollout.data, clone=True)

                    pbar.set_postfix({**metrics, "episodes": episode_count}, refresh=False)
                    for metric_name in ("policy_loss", "value_loss", "entropy", "ewc_loss"):
                        if metric_name in metrics:
                            continual_metrics.setdefault(game_name + f"_{metric_name}", []).append(
                                metrics[metric_name]
                            )
                    pbar.update(rollout.transition_count)

                if next_evaluation is not None and step >= next_evaluation:
                    if dqn_metrics is not None:
                        dqn_metrics.flush()
                    evaluation_env = AtariEnv(
                        game_name, seed=game_seeds[game_name], training=False, backend=env_backend
                    )
                    try:
                        rewards = run_evaluation_episodes(
                            agent, evaluation_env, eval_episodes, eval_max_steps
                        )
                    finally:
                        evaluation_env.close()
                    learning_evaluations.append(
                        {
                            "step": step,
                            "elapsed_seconds": time.perf_counter() - started_at,
                            "rewards": rewards,
                            "mean_raw_reward": sum(rewards) / len(rewards),
                        }
                    )
                    learning_history[game_name] = {
                        "task": game_name,
                        "stage": game_idx + 1,
                        "evaluation_steps": evaluation_steps[game_name],
                        "evaluations": learning_evaluations,
                    }
                    (exp_root / "learning.json").write_text(
                        json.dumps(learning_history, indent=2) + "\n", encoding="utf-8"
                    )
                    next_evaluation = next(evaluation_targets, None)

        finally:
            if dqn_metrics is not None:
                dqn_metrics.flush()
            pbar.close()
            if projection is not None:
                projection.close()
            env.close()
            if video_recorder is not None:
                video_recorder.close()

        if episode_rewards:
            continual_metrics.setdefault(game_name + "_episode_reward", []).extend(episode_rewards)

        stage_rewards = {}
        for eval_game in games[: game_idx + 1]:
            eval_task_id = eval_game
            if isinstance(agent, EWCWrapper) and hasattr(agent.agent, "set_task"):
                agent.set_task(eval_task_id)
            elif hasattr(agent, "set_task"):
                agent.set_task(eval_task_id)

            if (
                eval_game == game_name
                and learning_evaluations
                and learning_evaluations[-1]["step"] == stage_steps
            ):
                # Same weights, task, seed, episode count, and step limit as the final periodic evaluation.
                episode_eval_rewards = list(learning_evaluations[-1]["rewards"])
            else:
                eval_env = AtariEnv(
                    eval_game, training=False, seed=game_seeds[eval_game], backend=env_backend
                )
                try:
                    episode_eval_rewards = run_evaluation_episodes(
                        agent, eval_env, eval_episodes, eval_max_steps
                    )
                finally:
                    eval_env.close()
            stage_rewards[eval_game] = episode_eval_rewards
            avg_eval_reward = sum(episode_eval_rewards) / len(episode_eval_rewards)

            eval_history[eval_game].append((game_idx + 1, avg_eval_reward))
            print(
                f"[Eval] After task {game_name}, on {eval_game}: avg reward {avg_eval_reward:.2f}"
            )

        stage_episode_rewards.append(
            {
                "stage": game_idx + 1,
                "elapsed_seconds": time.perf_counter() - started_at,
                "rewards": stage_rewards,
            }
        )

        if use_ewc:
            sample_batch = None
            if algorithm == "dqn" and buffer.is_ready(batch_size):
                fisher_rng = np.random.default_rng([seed, game_idx])
                sample_batch = buffer.sample(min(batch_size, 100), rng=fisher_rng)
            elif algorithm == "ppo":
                sample_batch = last_rollout_data

            ewc_diagnostics[game_name] = agent.consolidate_weights(sample_batch)
            print(f"Weights consolidated for EWC: {ewc_diagnostics[game_name]}")

        if method == "gpm":
            agent.set_task(task_id)
            states = collect_gpm_states(
                agent,
                num_envs=num_envs,
                env_backend=env_backend,
                env_threads=env_threads,
                seed=seed + 20000 + game_idx * 1000,
                collection_steps=gpm_collection_steps,
                samples=gpm_samples,
            )
            subspaces = build_input_subspaces(
                gpm_backbone,
                states,
                threshold=gpm_threshold,
                seed=seed + 40000 + game_idx * 1000,
                batch_size=batch_size,
                previous=subspaces,
            )
            gpm_diagnostics[game_name] = {
                "boundary_collection_transitions": gpm_collection_steps,
                "boundary_num_envs": num_envs,
                "boundary_policy": "greedy" if algorithm == "dqn" else "stochastic",
                "sample_count": len(states),
                "projection": projection.metrics() if projection is not None else None,
                "layers": {
                    name: {key: value for key, value in entry.items() if not torch.is_tensor(value)}
                    for name, entry in subspaces.items()
                },
            }
            del states

        checkpoint = agent.checkpoint_state()
        checkpoint["training_stage"] = {
            "stage": game_idx + 1,
            "completed_tasks": games[: game_idx + 1],
            "task_steps": dict(zip(games[: game_idx + 1], budgets[: game_idx + 1], strict=True)),
            "seed": seed,
            "deterministic": deterministic,
            "evaluation": build_evaluation_report(eval_history, games),
            "stage_episode_rewards": list(stage_episode_rewards),
            "head_initialization_seeds": head_seeds,
            "purpose": "Task-boundary evaluation; not an exact training-resume snapshot",
        }
        if method == "gpm":
            checkpoint["gpm"] = {
                "subspaces": subspaces,
                "completed_tasks": games[: game_idx + 1],
                "threshold": gpm_threshold,
                "samples": gpm_samples,
                "collection_steps": gpm_collection_steps,
            }
        boundary_path = checkpoint_dir / (
            "final.pt" if game_idx == len(games) - 1 else f"stage-{game_idx + 1:02d}.pt"
        )
        torch.save(checkpoint, boundary_path)
        boundary_checkpoints.append(str(boundary_path.relative_to(exp_root)))
        (exp_root / "stage_evaluation.json").write_text(
            json.dumps(checkpoint["training_stage"], indent=2), encoding="utf-8"
        )

    metrics_plotter = MetricsPlotter()
    for name, values in continual_metrics.items():
        for v in values:
            metrics_plotter.add_metric(name, v)

    exp_root.mkdir(parents=True, exist_ok=True)

    metrics_output_path = exp_root / "figures" / "training_metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Continual metrics plot saved: {metrics_output_path}")

    eval_metrics_path = exp_root / "figures" / "forgetting_eval.png"
    plot_forgetting_curves(eval_history, eval_metrics_path)

    evaluation_report = {
        "checkpoint_sha256": file_digest(checkpoint_dir / "final.pt"),
        "algorithm": algorithm,
        "method": method,
        "initialization": "random",
        "games": games,
        "use_ewc": use_ewc,
        "ewc_lambda": ewc_lambda if use_ewc else None,
        "steps_per_game": steps_per_game if task_steps is None else None,
        "task_steps": dict(zip(games, budgets, strict=True)),
        "batch_size": batch_size,
        "num_envs": num_envs,
        "env_backend": env_backend,
        "env_threads": env_threads,
        "environment_protocol": observation_protocol(env_backend),
        "replay_sampler": "independent_pcg64_without_replacement" if algorithm == "dqn" else None,
        "replay_seeds": {game: [seed, index, 1] for index, game in enumerate(games)}
        if algorithm == "dqn"
        else None,
        "compile_ppo": compile_ppo if algorithm == "ppo" else None,
        "compile_dqn": compile_dqn if algorithm == "dqn" else None,
        "eval_episodes": eval_episodes,
        "eval_interval": eval_interval,
        "eval_points": eval_points,
        "evaluation_steps": evaluation_steps,
        "eval_max_steps": eval_max_steps,
        "seed": seed,
        "deterministic": deterministic,
        "boundary_checkpoints": boundary_checkpoints,
        "task_seeds": game_seeds,
        "head_initialization_seeds": head_seeds,
        "stage_episode_rewards": stage_episode_rewards,
        "ewc_diagnostics": ewc_diagnostics if use_ewc else None,
        "gpm_diagnostics": gpm_diagnostics if method == "gpm" else None,
        **build_evaluation_report(eval_history, games),
    }
    evaluation_path = exp_root / "training_summary.json"
    with evaluation_path.open("w", encoding="utf-8") as output_file:
        json.dump(evaluation_report, output_file, indent=2)
    print(f"Continual evaluation data saved: {evaluation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games", nargs="+", default=["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    )
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument("--method", choices=("finetune", "ewc", "gpm"))
    methods.add_argument(
        "--use-ewc", action="store_true", help="Compatibility alias for --method ewc"
    )
    parser.add_argument("--ewc-lambda", type=float, default=0.4, help="EWC regularization strength")
    parser.add_argument("--steps-per-game", type=int, default=50000)
    parser.add_argument(
        "--task-steps", type=int, nargs="+", help="Per-game budgets, overriding --steps-per-game"
    )
    parser.add_argument("--gpm-threshold", type=float, default=0.995)
    parser.add_argument("--gpm-samples", type=int, default=2048)
    parser.add_argument("--gpm-collection-steps", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-backend", choices=("sync", "async", "ale"), default="sync")
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--compile-ppo", action="store_true")
    parser.add_argument("--compile-dqn", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument(
        "--eval-points",
        type=int,
        default=0,
        help="Evaluation points per task budget; exclusive with --eval-interval",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Enable video recording (disabled by default)"
    )

    parser.add_argument("--output-dir", type=Path, help="Case directory; default: results/<case>")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing results after success"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_DIR / case_name(
        "continual",
        args.algorithm,
        method=args.method or ("ewc" if args.use_ewc else "finetune"),
        seed=args.seed,
    )
    with result_directory(output_dir, force=args.force) as staging:
        train_continual(
            output_dir=staging,
            games=args.games,
            algorithm=args.algorithm,
            use_ewc=args.use_ewc,
            ewc_lambda=args.ewc_lambda,
            steps_per_game=args.steps_per_game,
            batch_size=args.batch_size,
            num_envs=args.num_envs,
            env_backend=args.env_backend,
            env_threads=args.env_threads,
            compile_ppo=args.compile_ppo,
            compile_dqn=args.compile_dqn,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
            save_video=args.save_video,
            seed=args.seed,
            deterministic=args.deterministic,
            eval_interval=args.eval_interval,
            eval_points=args.eval_points,
            method=args.method,
            task_steps=args.task_steps,
            gpm_threshold=args.gpm_threshold,
            gpm_samples=args.gpm_samples,
            gpm_collection_steps=args.gpm_collection_steps,
        )
    print(f"Results saved to {output_dir}")
