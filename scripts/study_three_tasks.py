"""Single-seed Pong -> Breakout -> SpaceInvaders with PPO, original EWC and GPM."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch

from algorithms import EWCWrapper
from algorithms.subspace_projection import AdamSubspaceProjection, build_input_subspaces
from environments import AtariEnv, make_vector_atari_env
from scripts.study_gpm import (
    NEW,
    OLD,
    ROOT,
    evaluate,
    file_hash,
    new_agent,
    normalized_auc,
    source_hashes,
    train_stage,
    write_json,
)
from training import PPOCollector, PPOLearner, configure_ppo_runtime, flatten_rollout_data
from utils import seed_everything

TASKS = (OLD, NEW, "SpaceInvaders-v5")
METHODS = ("finetune", "ewc", "gpm")


def boundary_states(agent, config, seed):
    """Collect a frozen-policy boundary sample with a private RNG; never update PPO."""
    env = make_vector_atari_env(
        agent.current_task,
        config["num_envs"],
        backend=config["env_backend"],
        seed=seed,
        training=False,
    )
    devices = [agent.device.index or 0] if agent.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            collector = PPOCollector(
                env, PPOLearner(agent), rollout_length=config["rollout_length"]
            )
            chunks, completed = [], 0
            while completed < config["memory_collection_steps"]:
                rollout = collector.collect(config["memory_collection_steps"] - completed)
                chunks.append(flatten_rollout_data(rollout.data)["states"].cpu().clone())
                completed += rollout.transition_count
            states = torch.cat(chunks)
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(states), generator=generator)[: config["memory_capacity"]]
            return states[indices]
    finally:
        env.close()


def basis_diagnostics(subspaces):
    return {
        name: {key: value for key, value in entry.items() if not isinstance(value, torch.Tensor)}
        | {
            "remaining_dimension": entry["dimension"] - entry["rank"],
            "orthonormal_error": float(
                (entry["basis"].T @ entry["basis"] - torch.eye(entry["rank"])).abs().max()
            ),
        }
        for name, entry in subspaces.items()
    }


def run_study(settings, seed):
    source = ROOT / settings["anchor_dir"]
    original = json.loads((source / "manifest.json").read_text())
    if seed != original["seed"]:
        raise ValueError("Seed must match the frozen Pong anchor")
    config = original["config"].copy()
    if settings.get("smoke"):
        config.update(
            new_steps=1024,
            eval_interval=1024,
            eval_episodes=1,
            memory_collection_steps=1024,
            memory_capacity=256,
            compile_ppo=False,
        )
    configure_ppo_runtime(config["env_backend"])
    if not torch.cuda.is_available():
        raise RuntimeError("This frozen GPU experiment requires CUDA")
    device = "cuda"
    hashes = source_hashes()
    hashes["scripts/study_three_tasks.py"] = file_hash(Path(__file__))
    for name, expected in original["source_sha256"].items():
        if file_hash(source / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen snapshot changed: {name}")
        if (
            name.startswith(("algorithms/", "training/", "environments/", "utils/"))
            and name in hashes
            and hashes[name] != expected
        ):
            raise ValueError(f"Frozen PPO core changed: {name}")
    output = ROOT / settings["output_dir"] / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        settings=settings,
        protocol=config,
        tasks=TASKS,
        methods=METHODS,
        seed=seed,
        anchor_sha256=file_hash(source / "anchor.pt"),
        source_sha256=hashes,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
    )
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Manifest changed; use a new output directory")
    else:
        write_json(path, manifest)
        for name in hashes:
            snapshot = output / "source_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / name).read_bytes())
    anchor = torch.load(source / "anchor.pt", map_location="cpu", weights_only=True)
    # Evaluate future task with the same head initialization at every boundary.
    # Forking RNG avoids changing the original Breakout sampling path.
    for method in METHODS:
        destination = output / method
        if (destination / "summary.json").exists():
            continue
        destination.mkdir(exist_ok=True)
        gc.collect()
        torch.compiler.reset()
        torch.cuda.empty_cache()
        seed_everything(seed + 1000)
        agent = new_agent(device)
        agent.load_checkpoint_state(copy.deepcopy(anchor["agent"]))
        with torch.random.fork_rng(devices=[agent.device.index or 0]):
            torch.manual_seed(seed + 2000)
            env = AtariEnv(TASKS[2])
            try:
                agent.register_task(TASKS[2], int(env.action_space))
            finally:
                env.close()
        agent.set_task(NEW)
        train_agent = agent
        if method == "ewc":
            agent.set_task(OLD)
            train_agent = EWCWrapper(agent, ewc_lambda=config["ewc_lambda"])
            train_agent.consolidate_weights(anchor["last_rollout"])
            train_agent.set_task(NEW)
        subspaces = None
        if method == "gpm":
            subspaces = build_input_subspaces(
                agent.backbone,
                anchor["memory"]["tasks"][OLD][0],
                threshold=settings["threshold"],
                seed=seed + 40000,
                patches_per_observation=settings["patches_per_observation"],
            )
            torch.save(subspaces, destination / "subspaces_after_pong.pt")
        matrix = [
            {
                OLD: anchor["old_scores"][: config["eval_episodes"]],
                NEW: anchor["new_scores"][: config["eval_episodes"]],
                TASKS[2]: evaluate(agent, TASKS[2], config, seed + 12000),
            }
        ]
        stage_summaries = []
        for stage, task in enumerate(TASKS[1:], start=1):
            train_agent.set_task(task)
            if stage == 1:
                torch.set_rng_state(anchor["rng_cpu"])
                torch.cuda.set_rng_state_all(anchor["rng_cuda"])
            else:
                seed_everything(seed + 1000 * stage)
            # Record evaluates only deterministic actions; restore RNG before training.
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            stage_dir = destination / f"stage-{stage + 1}"
            stage_dir.mkdir(exist_ok=True)
            old_heads = {
                kind: {
                    t: copy.deepcopy(getattr(agent, kind)[t].state_dict()) for t in TASKS[:stage]
                }
                for kind in ("actors", "critics")
            }
            projection = (
                AdamSubspaceProjection(agent.optimizer, agent.backbone, subspaces)
                if method == "gpm"
                else None
            )
            history = []

            def record(steps, seconds, metrics):
                # Full stage-by-task matrix at boundaries; seen-task curves in between.
                tasks = TASKS if steps in (0, config["new_steps"]) else TASKS[: stage + 1]
                scores = (
                    matrix[-1]
                    if steps == 0
                    else {
                        t: evaluate(agent, t, config, seed + 10000 + 1000 * TASKS.index(t))
                        for t in tasks
                    }
                )
                history.append(
                    dict(
                        steps=steps,
                        training_seconds=seconds,
                        scores=scores,
                        ppo_metrics=metrics,
                        projection=projection.metrics() if projection else None,
                    )
                )
                write_json(stage_dir / "history.json", {"history": history})
                print(json.dumps({"method": method, "stage": stage + 1, **history[-1]}), flush=True)

            started = time.perf_counter()
            try:
                record(0, 0.0, {})
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                last_rollout, seconds = train_stage(
                    train_agent,
                    config,
                    seed + 1000 * stage,
                    config["new_steps"],
                    record=record,
                )
            finally:
                if projection:
                    projection.close()
            for kind, tasks in old_heads.items():
                for old_task, state in tasks.items():
                    for name, value in state.items():
                        torch.testing.assert_close(
                            getattr(agent, kind)[old_task].state_dict()[name], value, rtol=0, atol=0
                        )
            matrix.append(history[-1]["scores"])
            stage_summaries.append(
                dict(
                    task=task,
                    training_seconds=seconds,
                    wall_seconds=time.perf_counter() - started,
                    new_auc=normalized_auc(history, task),
                    old_heads_unchanged=True,
                    projection=projection.metrics() if projection else None,
                    subspaces=basis_diagnostics(subspaces) if subspaces else None,
                )
            )
            # Save exact boundary data and full EWC state for future continuation.
            torch.save(last_rollout, stage_dir / "last_rollout.pt")
            if method == "ewc":
                train_agent.consolidate_weights(last_rollout)
                train_agent.save(str(stage_dir / "final.pt"))
            else:
                torch.save({"agent": agent.checkpoint_state()}, stage_dir / "final.pt")
            if method == "gpm":
                boundary_start = time.perf_counter()
                states = boundary_states(agent, config, seed + 20000 + stage * 1000)
                subspaces = build_input_subspaces(
                    agent.backbone,
                    states,
                    threshold=settings["threshold"],
                    seed=seed + 40000 + stage * 1000,
                    patches_per_observation=settings["patches_per_observation"],
                    previous=subspaces,
                )
                stage_summaries[-1]["boundary_collection_and_basis_seconds"] = (
                    time.perf_counter() - boundary_start
                )
                stage_summaries[-1]["boundary_collection_transitions"] = config[
                    "memory_collection_steps"
                ]
                torch.save(subspaces, stage_dir / "subspaces.pt")
                write_json(stage_dir / "subspace_diagnostics.json", basis_diagnostics(subspaces))
            write_json(
                destination / "progress.json", dict(score_matrix=matrix, stages=stage_summaries)
            )
        write_json(destination / "summary.json", dict(score_matrix=matrix, stages=stage_summaries))
        agent = train_agent = projection = None
    write_json(output / "completion.json", {"status": "completed", "methods": METHODS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run_study(json.loads(args.config.read_text()), args.seed)


if __name__ == "__main__":
    main()
