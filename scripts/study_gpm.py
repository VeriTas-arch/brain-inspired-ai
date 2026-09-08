"""Single-seed GPM qualification from the frozen policy-replay Pong boundary."""

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from algorithms import MultiHeadPPOAgent
from algorithms.subspace_projection import AdamSubspaceProjection, build_input_subspaces
from environments import AtariEnv, make_vector_atari_env
from training import (
    PPOCollector,
    PPOLearner,
    configure_ppo_runtime,
    flatten_rollout_data,
    run_evaluation_episodes,
)
from utils import seed_everything

OLD, NEW = "Pong-v5", "Breakout-v5"
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    paths = [ROOT / "scripts/study_gpm.py", ROOT / "scripts/run_experiments.py"]
    for directory in ("algorithms", "training", "environments", "utils"):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def new_agent(device: str) -> MultiHeadPPOAgent:
    return MultiHeadPPOAgent(state_dim=4, device=device)


def evaluate(agent, task: str, config: dict, seed: int) -> list[float]:
    previous = agent.current_task
    agent.set_task(task)
    environment = AtariEnv(task, seed=seed, training=False)
    try:
        return run_evaluation_episodes(
            agent, environment, config["eval_episodes"], config["eval_max_steps"]
        )
    finally:
        environment.close()
        agent.set_task(previous)


def policy_diagnostics(agent, probe: dict, batch_size: int) -> dict:
    states, teacher = probe["tasks"][OLD]
    kl, matches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            target = teacher[start : start + batch_size].to(agent.device).log_softmax(-1)
            student = agent.policy_logits(
                states[start : start + batch_size].to(agent.device), OLD
            ).log_softmax(-1)
            kl += (target.exp() * (target - student)).sum().item()
            matches += (target.argmax(-1) == student.argmax(-1)).sum().item()
    return {"old_policy_kl": kl / len(states), "action_agreement": matches / len(states)}


def train_stage(agent, config: dict, seed: int, steps: int, *, regularizer=None, record=None):
    """Protocol scheduling only; collection, GAE and optimization stay in the runtime."""
    task = getattr(agent, "agent", agent).current_task
    environment = make_vector_atari_env(
        task, config["num_envs"], backend=config["env_backend"], seed=seed
    )
    learner = PPOLearner(
        agent,
        update_epochs=config["update_epochs"],
        minibatch_size=config["minibatch_size"],
        compile_policy=config["compile_ppo"],
        regularizer=regularizer,
    )
    collector = PPOCollector(environment, learner, rollout_length=config["rollout_length"])
    completed, training_seconds = 0, 0.0
    try:
        while completed < steps:
            remaining = min(
                steps - completed, config["eval_interval"] - completed % config["eval_interval"]
            )
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            rollout = collector.collect(remaining)
            if completed == 0:
                digest = hashlib.sha256()
                for field in ("states", "actions", "rewards", "dones"):
                    digest.update(rollout.data[field].cpu().contiguous().numpy().tobytes())
                print(
                    json.dumps({"task": task, "first_rollout_sha256": digest.hexdigest()}),
                    flush=True,
                )
            metrics = learner.update(rollout)
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.perf_counter() - started
            completed += rollout.transition_count
            if completed % config["eval_interval"] == 0 or completed == steps:
                print(
                    json.dumps(
                        {
                            "task": task,
                            "steps": completed,
                            "train_seconds": training_seconds,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
                if record is not None:
                    record(completed, training_seconds, metrics)
        return flatten_rollout_data(rollout.data, clone=True), training_seconds
    finally:
        environment.close()


def normalized_auc(history: list[dict], task: str) -> float:
    steps = np.array([row["steps"] for row in history])
    means = np.array([np.mean(row["scores"][task]) for row in history])
    return float(np.trapezoid(means, steps) / (steps[-1] - steps[0]))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_study(settings: dict, seed: int) -> None:
    source = ROOT / settings["anchor_dir"]
    original = json.loads((source / "manifest.json").read_text())
    if seed != original["seed"]:
        raise ValueError("The seed must match the frozen anchor")
    config = original["config"].copy()
    if settings.get("smoke", False):
        config.update(new_steps=1024, eval_interval=1024, eval_episodes=1)
    configure_ppo_runtime(config["env_backend"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = ROOT / settings["output_dir"] / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    hashes["scripts/study_gpm.py"] = file_hash(Path(__file__))
    for name, expected in original["source_sha256"].items():
        if file_hash(source / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen anchor source snapshot changed: {name}")
    changed = [
        name
        for name, value in original["source_sha256"].items()
        if name.startswith(("algorithms/", "training/", "environments/", "utils/"))
        and name in hashes
        and hashes[name] != value
    ]
    if changed:
        raise ValueError(f"Frozen PPO/evaluation sources changed: {changed}")
    manifest = {
        "settings": settings,
        "protocol": config,
        "seed": seed,
        "anchor_sha256": file_hash(source / "anchor.pt"),
        "source_sha256": hashes,
        "device": device,
        "torch": torch.__version__,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Study code/config changed; use a new output directory")
    else:
        write_json(manifest_path, manifest)
        for name in hashes:
            snapshot = output / "source_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / name).read_bytes())

    anchor = torch.load(source / "anchor.pt", map_location="cpu", weights_only=True)
    subspace_path = output / "subspaces.pt"
    if not subspace_path.exists():
        agent = new_agent(device)
        agent.load_checkpoint_state(copy.deepcopy(anchor["agent"]))
        started = time.perf_counter()
        states = anchor["memory"]["tasks"][OLD][0]
        subspaces = build_input_subspaces(
            agent.backbone,
            states,
            threshold=settings["threshold"],
            seed=seed + 40000,
            batch_size=config["minibatch_size"],
            patches_per_observation=settings["patches_per_observation"],
        )
        torch.save(subspaces, subspace_path)
        diagnostics = {
            name: {
                key: value for key, value in entry.items() if not isinstance(value, torch.Tensor)
            }
            | {
                "orthonormal_error": float(
                    (entry["basis"].T @ entry["basis"] - torch.eye(entry["rank"])).abs().max()
                ),
            }
            for name, entry in subspaces.items()
        }
        write_json(
            output / "subspace_diagnostics.json",
            {"construction_seconds": time.perf_counter() - started, "layers": diagnostics},
        )
        print(json.dumps({"subspaces": diagnostics}), flush=True)
        del agent, subspaces
    subspaces = torch.load(subspace_path, map_location="cpu", weights_only=True)
    memory_bytes = sum(
        entry[key].numel() * entry[key].element_size()
        for entry in subspaces.values()
        for key in ("basis",)
    )
    methods = ["gpm"]
    for method in methods:
        destination = output / method
        destination.mkdir(exist_ok=True)
        if (destination / "summary.json").exists():
            continue
        gc.collect()
        torch.compiler.reset()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        seed_everything(seed + 1000)
        agent = new_agent(device)
        agent.load_checkpoint_state(copy.deepcopy(anchor["agent"]))
        projection = AdamSubspaceProjection(
            agent.optimizer,
            agent.backbone,
            subspaces,
        )
        probe = anchor["probe"]
        history = []

        def record(steps, seconds, metrics):
            scores = {
                task: anchor[key][: config["eval_episodes"]]
                if steps == 0
                else evaluate(agent, task, config, eval_seed)
                for task, key, eval_seed in (
                    (OLD, "old_scores", seed + 10000),
                    (NEW, "new_scores", seed + 11000),
                )
            }
            history.append(
                {
                    "steps": steps,
                    "training_seconds": seconds,
                    "scores": scores,
                    "reused_boundary_scores": steps == 0,
                    "ppo_metrics": metrics,
                    "projection": projection.metrics(),
                    **policy_diagnostics(agent, probe, config["minibatch_size"]),
                }
            )
            write_json(destination / "history.json", {"history": history})
            print(json.dumps({"branch": method, **history[-1]}), flush=True)

        wall_start = time.perf_counter()
        try:
            record(0, 0.0, {})
            torch.set_rng_state(anchor["rng_cpu"])
            if device == "cuda":
                torch.cuda.set_rng_state_all(anchor["rng_cuda"])
            _, seconds = train_stage(agent, config, seed + 1000, config["new_steps"], record=record)
            state = agent.checkpoint_state()
            # Old heads are not trained; only their shared input representation can drift.
            for kind in ("actors", "critics"):
                for name, value in anchor["agent"][kind][OLD].items():
                    torch.testing.assert_close(state[kind][OLD][name].cpu(), value, rtol=0, atol=0)
            torch.save({"agent": state, "method": method}, destination / "final.pt")
            final = history[-1]
            write_json(
                destination / "summary.json",
                {
                    "branch": method,
                    "training_seconds": seconds,
                    "wall_seconds_including_evaluation": time.perf_counter() - wall_start,
                    "new_auc_mean_reward": normalized_auc(history, NEW),
                    "final_old_score": float(np.mean(final["scores"][OLD])),
                    "final_new_score": float(np.mean(final["scores"][NEW])),
                    "old_score_change": float(
                        np.mean(final["scores"][OLD]) - np.mean(history[0]["scores"][OLD])
                    ),
                    "old_policy_kl": final["old_policy_kl"],
                    "action_agreement": final["action_agreement"],
                    "projection_memory_bytes": memory_bytes,
                    "projection": projection.metrics(),
                    "old_heads_unchanged": True,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                    if device == "cuda"
                    else 0,
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
                    if device == "cuda"
                    else 0,
                },
            )
        finally:
            projection.close()
        agent = projection = probe = record = None
    write_json(output / "completion.json", {"status": "completed", "methods": methods})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_study(json.loads(args.config.read_text()), args.seed)


if __name__ == "__main__":
    main()
