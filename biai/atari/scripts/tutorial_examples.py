"""Short, testable examples used by the teaching notebook."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

from biai.atari.algorithms import DQNAgent, EWCWrapper
from biai.atari.algorithms.ppo import generalized_advantage_estimate
from biai.paths import ASSETS_DIR, RESULTS_DIR


def runtime_summary() -> dict[str, object]:
    """Return the runtime information shown at the start of the notebook."""
    cuda_available = torch.cuda.is_available()
    cuda_usable = False
    cuda_error = None
    if cuda_available:
        try:
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            cuda_usable = True
        except RuntimeError as error:
            cuda_error = str(error).splitlines()[0]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchrl": version("torchrl"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_usable": cuda_usable,
        "cuda_error": cuda_error,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _dqn_batch(batch_size: int, action_dim: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "states": torch.randint(
            0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8, generator=generator
        ),
        "actions": torch.randint(0, action_dim, (batch_size,), generator=generator),
        "rewards": torch.randn(batch_size, generator=generator),
        "next_states": torch.randint(
            0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8, generator=generator
        ),
        "dones": torch.zeros(batch_size),
    }


def run_dqn_update_demo(seed: int = 7, batch_size: int = 4) -> dict[str, object]:
    """Run one real DQN update on a small synthetic transition batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(seed)
    action_dim = 4
    agent = DQNAgent(state_dim=4, action_dim=action_dim, device="cpu")
    batch = _dqn_batch(batch_size, action_dim, seed)
    metrics = agent.update(batch)
    return {
        "batch_shapes": {name: tuple(tensor.shape) for name, tensor in batch.items()},
        **metrics,
    }


def run_gae_demo() -> dict[str, list[float]]:
    """Compute a small, deterministic GAE example using the maintained implementation."""
    advantages, returns = generalized_advantage_estimate(
        rewards=torch.tensor([0.0, 0.0, 1.0]),
        values=torch.tensor([0.2, 0.3, 0.4]),
        dones=torch.tensor([0.0, 0.0, 1.0]),
        next_value=torch.tensor(0.0),
        gamma=0.99,
        gae_lambda=0.95,
    )
    return {
        "advantages": advantages.tolist(),
        "returns": returns.tolist(),
    }


def run_ewc_penalty_demo(seed: int = 7, batch_size: int = 2) -> dict[str, float]:
    """Show that the maintained EWC penalty grows after moving consolidated weights."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(seed)
    action_dim = 4
    agent = DQNAgent(state_dim=4, action_dim=action_dim, device="cpu")
    wrapper = EWCWrapper(agent, ewc_lambda=0.4)
    batch = _dqn_batch(batch_size, action_dim, seed)
    wrapper.consolidate_weights(batch)
    penalty_at_reference = wrapper.compute_ewc_loss().item()

    with torch.no_grad():
        for parameter in agent.network.parameters():
            parameter.add_(0.01)
    penalty_after_move = wrapper.compute_ewc_loss().item()
    return {
        "penalty_at_reference": penalty_at_reference,
        "penalty_after_move": penalty_after_move,
    }


def main() -> None:
    """Run the same short examples exposed by the notebook."""
    print("Runtime:", runtime_summary())
    print("DQN update:", run_dqn_update_demo())
    print("GAE:", run_gae_demo())
    print("EWC penalty:", run_ewc_penalty_demo())


if __name__ == "__main__":
    main()


TEACHING_RESULTS = RESULTS_DIR
TEACHING_ASSETS = ASSETS_DIR
REFERENCE_RESULTS = TEACHING_ASSETS / "reference_results.json"


def load_teaching_results(results_dir: Path = REFERENCE_RESULTS) -> dict:
    """Read case records without requiring local models or source copies."""
    results_dir = Path(results_dir)
    if results_dir.is_file():
        return json.loads(results_dir.read_text())
    result = {
        "training": {},
        "configurations": {},
        "runs": {},
        "evaluation_runs": {},
        "models": [],
        "continual": {},
        "learning_curves": {},
    }
    for config_path in sorted(results_dir.glob("*/config.json")):
        directory = config_path.parent
        case = directory.name
        if case.startswith(".") or case in {"smoke", "performance"}:
            continue
        config = json.loads(config_path.read_text())
        result["configurations"][case] = config
        evaluation_record = directory / "evaluation/run.json"
        if evaluation_record.exists():
            result["evaluation_runs"][case] = json.loads(evaluation_record.read_text())
        result["runs"][case] = json.loads((directory / "run.json").read_text())
        summary = json.loads((directory / "training_summary.json").read_text())
        result["training"][case] = {
            key: summary[key]
            for key in ("total_steps", "task_steps", "num_envs", "environment_protocol")
            if key in summary
        }
        metrics = json.loads((directory / "evaluation/evaluation.json").read_text())
        result["models"].append(
            {
                "name": case,
                "checkpoint": f"{case}/checkpoints/final.pt",
                "sha256": summary["checkpoint_sha256"],
                "metrics_path": f"{case}/evaluation/evaluation.json",
                "metrics": metrics,
            }
        )
        if config["protocol"] == "continual":
            result["continual"][case] = summary
        learning = directory / "learning.json"
        if learning.is_file():
            result["learning_curves"][f"{case}/learning.json"] = json.loads(learning.read_text())
    return result


def teaching_results(case: str, results_path: Path = REFERENCE_RESULTS) -> str:
    """Render final means or stage trajectories from the current teaching cases."""
    data = load_teaching_results(results_path)
    lines = ["| 算法 | 游戏 | 评估均分／阶段轨迹 |", "|---|---|---|"]
    if case in {"single", "multitask"}:
        for record in data["models"]:
            metrics = record["metrics"]
            if metrics["mode"] != case:
                continue
            games = metrics["games"] if case == "multitask" else {metrics["game"]: metrics}
            for game, scores in games.items():
                lines.append(
                    f"| {metrics['algorithm'].upper()} | {game} | {scores['avg_reward']:g} |"
                )
    elif case in {"finetune", "ewc", "gpm"}:
        for summary in data["continual"].values():
            if summary["method"] != case:
                continue
            for game in summary["games"]:
                values = [row["scores"][game] for row in summary["score_matrix"]]
                trajectory = " → ".join(f"{value:g}" for value in values if value is not None)
                lines.append(f"| {summary['algorithm'].upper()} | {game} | {trajectory} |")
    else:
        raise ValueError(f"Unknown teaching case: {case}")
    return "\n".join(lines)


def matched_budget_results(results_dir: Path, seed: int = 0) -> str:
    """Compare completed local cases only after checking their per-game budgets."""
    from biai.atari.scripts.run_experiments import build_teaching_jobs

    expected = {
        job.case_id for job in build_teaching_jobs(phase="evaluate", seed=seed, matched_budget=True)
    }
    missing = sorted(
        case
        for case in expected
        if not (Path(results_dir) / case / "evaluation/evaluation.json").is_file()
    )
    if missing:
        return f"预算匹配实验尚未完成：还缺 {len(missing)}/{len(expected)} 个案例的独立评估。"
    data = load_teaching_results(results_dir)
    budgets, rows = {}, []
    for record in data["models"]:
        case = record["name"]
        if case not in expected:
            continue
        config, metrics = data["configurations"][case], record["metrics"]
        training = data["training"][case]
        games = {metrics["game"]: metrics} if metrics["mode"] == "single" else metrics["games"]
        for game, scores in games.items():
            steps = (
                training["total_steps"]
                if metrics["mode"] == "single"
                else training["task_steps"][game]
            )
            key = (metrics["algorithm"], game)
            if key in budgets and budgets[key] != steps:
                raise ValueError(
                    f"Mismatched training budgets for {key}: {budgets[key]} vs {steps}"
                )
            budgets[key] = steps
            protocol = (
                config.get("method", metrics["mode"])
                if metrics["mode"] == "continual"
                else metrics["mode"]
            )
            rows.append((metrics["algorithm"].upper(), game, protocol, steps, scores["avg_reward"]))
    lines = [
        "| 算法 | 游戏 | 训练方式 | 该游戏训练步数 | 独立评估均分 |",
        "|---|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {algorithm} | {game} | {protocol} | {steps} | {score:g} |"
        for algorithm, game, protocol, steps, score in sorted(rows)
    )
    return "\n".join(lines)
