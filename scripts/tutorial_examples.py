"""Short, testable examples used by the teaching notebook."""

from __future__ import annotations

import platform
from importlib.metadata import version

import torch

from algorithms import DQNAgent, EWCWrapper
from algorithms.ppo import generalized_advantage_estimate


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


def teaching_results(case: str) -> str:
    """Render the committed evaluation snapshot, without requiring local training outputs."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "teaching.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    records = [record for record in snapshot["records"] if record["case"] == case]
    if not records:
        raise ValueError(f"Unknown teaching case: {case}")
    lines = [
        "| 算法 | 游戏 | 评估均分／阶段轨迹 | 证据状态 |",
        "|---|---|---|---|",
    ]
    for record in records:
        data = record["data"]
        if case == "single":
            scores = {
                record["game"]: [e["mean_raw_reward"] for e in data["evaluations"]]
                if "evaluations" in data
                else [data["avg_reward"]]
            }
        elif case == "multitask":
            scores = {game: [entry["avg_reward"]] for game, entry in data["games"].items()}
        else:
            scores = {
                game: [
                    stage["scores"][game]
                    for stage in data["score_matrix"]
                    if stage["scores"][game] is not None
                ]
                for game in data["games"]
            }
        for game, values in scores.items():
            trajectory = " → ".join(f"{value:g}" for value in values)
            lines.append(
                f"| {record['algorithm'].upper()} | {game} | {trajectory} | {record['status']} |"
            )
    return "\n".join(lines)
