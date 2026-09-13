"""Evaluate trained Atari agents (single-task or continual)."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from biai.atari.algorithms import (
    DQNAgent,
    EWCWrapper,
    MultiHeadDQNAgent,
    MultiHeadPPOAgent,
    PPOAgent,
)
from biai.atari.environments import AtariEnv, observation_protocol
from biai.atari.training import (
    DEFAULT_MAX_EPISODE_STEPS,
    VideoRecorder,
    run_evaluation_episodes,
    seed_everything,
)
from biai.atari.training.results import (
    check_result_path,
    file_digest,
    result_directory,
    run_metadata,
)


def _evaluation_backend(agent):
    inner = agent.agent if isinstance(agent, EWCWrapper) else agent
    protocol = getattr(inner, "environment_protocol", "gymnasium_wrappers_v1")
    if protocol not in {"gymnasium_wrappers_v1", "ale_native_v1"}:
        raise ValueError(f"Unknown checkpoint observation protocol: {protocol}")
    return "ale" if protocol == "ale_native_v1" else "sync"


def _infer_eval_dir_from_model_path(model_path: str) -> Path:
    """Use one stable evaluation directory beside the model checkpoints."""
    model = Path(model_path)
    case = model.parent.parent if model.parent.name == "checkpoints" else model.parent
    return case / "evaluation"


def _prepare_output_dir(output_dir: Path | None, model_path: str) -> Path:
    """Create and return the directory used by evaluation artifacts."""
    resolved = output_dir or _infer_eval_dir_from_model_path(model_path)
    check_result_path(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "figures").mkdir(exist_ok=True)
    (resolved / "videos").mkdir(exist_ok=True)
    return resolved


def _set_agent_eval(agent) -> None:
    """Put all neural modules owned by an agent into evaluation mode."""
    inner_agent = agent.agent if isinstance(agent, EWCWrapper) else agent
    module_names = (
        "network",
        "backbone",
        "target_network",
        "target_backbone",
        "actor",
        "critic",
        "heads",
        "target_heads",
        "actors",
        "critics",
    )
    for name in module_names:
        module = getattr(inner_agent, name, None)
        if isinstance(module, torch.nn.Module):
            module.eval()


def _plot_single_results(result: dict, output_dir: Path) -> Path:
    """Plot episode rewards for a single-task evaluation."""
    game = result["game"]
    algorithm = result["algorithm"]
    rewards = result["rewards"]
    plot_path = output_dir / "figures" / f"{game.replace('/', '_')}_{algorithm}_eval_rewards.png"

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o")
    plt.title(f"{game} ({algorithm}) Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def _plot_multi_game_results(result: dict, output_dir: Path) -> Path:
    """Plot each game's average raw reward without cross-game aggregation."""
    mode = result["mode"]
    algorithm = result["algorithm"]
    games = sorted(result["games"])
    avg_rewards = [result["games"][game]["avg_reward"] for game in games]
    plot_path = output_dir / "figures" / f"{mode}_{algorithm}_avg_rewards.png"

    plt.figure(figsize=(6, 4))
    plt.bar(games, avg_rewards)
    plt.title(f"{mode.title()} {algorithm.upper()} Raw Reward per Game")
    plt.xlabel("Game")
    plt.ylabel("Average Raw Reward")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def _record_example_video(
    agent,
    game: str,
    output_path: Path,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    seed: int = 0,
) -> None:
    """Record one deterministic evaluation episode."""
    env = AtariEnv(
        game, render_mode="rgb_array", training=False, seed=seed, backend=_evaluation_backend(agent)
    )
    try:
        if hasattr(agent, "set_task"):
            agent.set_task(game)
        with VideoRecorder(str(output_path), fps=30) as recorder:
            run_evaluation_episodes(
                agent,
                env,
                1,
                max_steps,
                frame_callback=lambda: recorder.add_frame(env.env.render()),
            )
    finally:
        env.close()


def evaluate_single(
    model_path: str,
    game: str,
    algorithm: str,
    episodes: int,
    max_steps: int,
    output_dir: Path | None = None,
    seed: int = 0,
    deterministic: bool = False,
    compile_dqn: bool = False,
    compile_ppo: bool = False,
) -> dict:
    """Evaluate a single-task agent checkpoint on one game."""
    if compile_ppo and algorithm != "ppo":
        raise ValueError("compile_ppo requires PPO")
    if compile_dqn and algorithm != "dqn":
        raise ValueError("compile_dqn requires DQN")
    seed_everything(seed, deterministic=deterministic)
    output_dir = _prepare_output_dir(output_dir, model_path)
    env = AtariEnv(game, render_mode=None, training=False, seed=seed)
    try:
        if algorithm == "dqn":
            agent = DQNAgent(state_dim=4, action_dim=env.action_space)
        else:
            agent = PPOAgent(state_dim=4, action_dim=env.action_space)
        agent.load(model_path)
        if _evaluation_backend(agent) == "ale":
            env.close()
            env = AtariEnv(game, training=False, seed=seed, backend="ale")
        if compile_dqn or compile_ppo:
            agent.configure_runtime(compile_enabled=True)
        _set_agent_eval(agent)
        episode_rewards = run_evaluation_episodes(agent, env, episodes, max_steps)
    finally:
        env.close()

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"[Single] {game} ({algorithm}) - Episodes: {episodes}")
    print(
        f"  Avg reward: {avg_reward:.2f}, Min: {min(episode_rewards):.2f}, "
        f"Max: {max(episode_rewards):.2f}"
    )
    result = {
        "mode": "single",
        "deterministic": deterministic,
        "compile_dqn": compile_dqn if algorithm == "dqn" else None,
        "compile_ppo": compile_ppo if algorithm == "ppo" else None,
        "environment_protocol": observation_protocol(_evaluation_backend(agent)),
        "game": game,
        "algorithm": algorithm,
        "episodes": episodes,
        "seed": seed,
        "rewards": episode_rewards,
        "avg_reward": avg_reward,
    }

    plot_path = _plot_single_results(result, output_dir)
    print(f"  Reward plot saved to: {plot_path}")
    video_path = output_dir / "videos" / f"{game.replace('/', '_')}_{algorithm}_eval_gameplay.mp4"
    _record_example_video(agent, game, video_path, max_steps=max_steps, seed=seed)
    print(f"  Example gameplay video saved to: {video_path}")
    return result


def _build_multihead_agent(algorithm: str, games: list[str], use_ewc: bool, ewc_lambda: float):
    """Rebuild a multi-head agent architecture for evaluation."""
    if algorithm == "dqn":
        base_agent = MultiHeadDQNAgent(state_dim=4)
    else:
        base_agent = MultiHeadPPOAgent(state_dim=4)

    for game in games:
        env = AtariEnv(game, render_mode=None, training=False)
        try:
            base_agent.register_task(game, env.action_space)
        finally:
            env.close()

    if use_ewc:
        return EWCWrapper(base_agent, ewc_lambda=ewc_lambda)
    return base_agent


def _evaluate_games(
    agent,
    games: list[str],
    mode: str,
    algorithm: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    """Evaluate a multi-head agent independently on every requested game."""
    results = {
        "mode": mode,
        "algorithm": algorithm,
        "games": {},
        "episodes": episodes,
        "seed": seed,
    }
    label = "Multi-task" if mode == "multitask" else "Continual"

    for game_index, game in enumerate(games):
        print(f"\n[{label}] Evaluating on {game} ...")
        env = AtariEnv(
            game,
            render_mode=None,
            training=False,
            seed=seed + game_index,
            backend=_evaluation_backend(agent),
        )
        try:
            agent.set_task(game)
            episode_rewards = run_evaluation_episodes(agent, env, episodes, max_steps)
        finally:
            env.close()

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(
            f"  Avg reward: {avg_reward:.2f}, Min: {min(episode_rewards):.2f}, "
            f"Max: {max(episode_rewards):.2f}"
        )
        results["games"][game] = {
            "rewards": episode_rewards,
            "avg_reward": avg_reward,
        }
    return results


def _evaluate_multihead_checkpoint(
    model_path: str,
    games: list[str],
    algorithm: str,
    episodes: int,
    max_steps: int,
    mode: str,
    use_ewc: bool,
    ewc_lambda: float,
    output_dir: Path | None,
    seed: int,
    deterministic: bool = False,
    compile_dqn: bool = False,
    compile_ppo: bool = False,
) -> dict:
    """Shared evaluation workflow for continual and joint multi-task checkpoints."""
    label = "multi-task" if mode == "multitask" else "continual"
    print(f"Loading {label} agent from {model_path}...")
    if compile_ppo and algorithm != "ppo":
        raise ValueError("compile_ppo requires PPO")
    if compile_dqn and algorithm != "dqn":
        raise ValueError("compile_dqn requires DQN")
    seed_everything(seed, deterministic=deterministic)
    output_dir = _prepare_output_dir(output_dir, model_path)
    agent = _build_multihead_agent(algorithm, games, use_ewc, ewc_lambda)
    agent.load(model_path)
    if compile_dqn or compile_ppo:
        agent.configure_runtime(compile_enabled=True)
    _set_agent_eval(agent)

    results = _evaluate_games(agent, games, mode, algorithm, episodes, max_steps, seed)
    results["environment_protocol"] = observation_protocol(_evaluation_backend(agent))
    results["deterministic"] = deterministic
    results["compile_ppo"] = compile_ppo if algorithm == "ppo" else None
    results["compile_dqn"] = compile_dqn if algorithm == "dqn" else None
    plot_path = _plot_multi_game_results(results, output_dir)
    print(f"\n[{label.title()}] Raw reward plot saved to: {plot_path}")

    for game_index, game in enumerate(games):
        video_path = (
            output_dir / "videos" / f"{mode}_{algorithm}_{game.replace('/', '_')}_eval_gameplay.mp4"
        )
        _record_example_video(
            agent,
            game,
            video_path,
            max_steps=max_steps,
            seed=seed + game_index,
        )
        print(f"[{label.title()}] Example gameplay video for {game} saved to: {video_path}")
    return results


def evaluate_continual(
    model_path: str,
    games: list[str],
    algorithm: str,
    episodes: int,
    max_steps: int,
    use_ewc: bool,
    ewc_lambda: float,
    output_dir: Path | None = None,
    seed: int = 0,
    deterministic: bool = False,
    compile_dqn: bool = False,
    compile_ppo: bool = False,
) -> dict:
    """Evaluate a continual-learning checkpoint on each game."""
    return _evaluate_multihead_checkpoint(
        model_path,
        games,
        algorithm,
        episodes,
        max_steps,
        "continual",
        use_ewc,
        ewc_lambda,
        output_dir,
        seed,
        deterministic,
        compile_dqn,
        compile_ppo,
    )


def evaluate_multitask(
    model_path: str,
    games: list[str],
    algorithm: str,
    episodes: int,
    max_steps: int,
    output_dir: Path | None = None,
    seed: int = 0,
    deterministic: bool = False,
    compile_dqn: bool = False,
    compile_ppo: bool = False,
) -> dict:
    """Evaluate a jointly trained multi-task checkpoint on each game."""
    return _evaluate_multihead_checkpoint(
        model_path,
        games,
        algorithm,
        episodes,
        max_steps,
        "multitask",
        False,
        0.0,
        output_dir,
        seed,
        deterministic,
        compile_dqn,
        compile_ppo,
    )


def main() -> None:
    """Parse command-line arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained Atari agents")
    parser.add_argument("--mode", choices=["single", "continual", "multitask"], default="single")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--game", help="Game name for single mode (e.g., Pong-v5)")
    parser.add_argument("--games", nargs="*", help="List of games for continual/multitask mode")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per game")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Maximum agent steps per episode; incomplete episodes are rejected",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile-dqn", action="store_true")
    parser.add_argument("--compile-ppo", action="store_true")
    parser.add_argument("--ewc", action="store_true", help="Use EWC wrapper in continual mode")
    parser.add_argument("--ewc-lambda", type=float, default=0.4, help="EWC lambda (if --ewc)")
    parser.add_argument("--json-out", help="Optional path to write JSON results")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing evaluation after success"
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.json_out).parent if args.json_out else _infer_eval_dir_from_model_path(args.model)
    )
    output_path = Path(args.json_out) if args.json_out else output_dir / "evaluation.json"
    check_result_path(output_dir)
    if Path(args.model).resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Evaluation output must be separate from the model directory")
    metadata = {
        **run_metadata(),
        "checkpoint": os.path.relpath(Path(args.model).resolve(), output_dir.resolve()),
        "checkpoint_sha256": file_digest(Path(args.model)),
    }
    with result_directory(output_dir, force=args.force) as staging:
        if args.mode == "single":
            if not args.game:
                raise SystemExit("--game is required for single mode")
            result = evaluate_single(
                args.model,
                args.game,
                args.algorithm,
                args.episodes,
                args.max_steps,
                staging,
                args.seed,
                deterministic=args.deterministic,
                compile_dqn=args.compile_dqn,
                compile_ppo=args.compile_ppo,
            )
        elif args.mode == "multitask":
            if not args.games:
                raise SystemExit("--games is required for multitask mode")
            result = evaluate_multitask(
                args.model,
                args.games,
                args.algorithm,
                args.episodes,
                args.max_steps,
                staging,
                args.seed,
                deterministic=args.deterministic,
                compile_dqn=args.compile_dqn,
                compile_ppo=args.compile_ppo,
            )
        else:
            if not args.games:
                raise SystemExit("--games is required for continual mode")
            result = evaluate_continual(
                args.model,
                args.games,
                args.algorithm,
                args.episodes,
                args.max_steps,
                args.ewc,
                args.ewc_lambda,
                staging,
                args.seed,
                deterministic=args.deterministic,
                compile_dqn=args.compile_dqn,
                compile_ppo=args.compile_ppo,
            )

        with (staging / output_path.name).open("x", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
        (staging / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
