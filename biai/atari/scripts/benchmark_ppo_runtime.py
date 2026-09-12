"""Benchmark the maintained PPO runtime without changing its minibatch size."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from biai.atari.algorithms import PPOAgent
from biai.atari.environments import make_vector_atari_env
from biai.atari.training import PPOCollector, PPOLearner, configure_ppo_runtime, seed_everything


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing and PyTorch allocator measurements for one runtime configuration."""

    backend: str
    compiled: bool
    transitions: int
    seconds: float
    collection_seconds: float
    update_seconds: float
    warmup_seconds: float
    transitions_per_second: float
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None


def benchmark_environment(environment, *, transitions, warmup_transitions, seed, backend):
    """Time only simulation, preprocessing, and delivery, with pre-generated actions."""
    environment.reset()
    actions = torch.from_numpy(
        np.random.default_rng(seed).integers(
            environment.action_space,
            size=((transitions + warmup_transitions) // environment.num_envs, environment.num_envs),
        )
    )
    warmup_steps = warmup_transitions // environment.num_envs
    warmup_start = time.perf_counter()
    for step, action in enumerate(actions):
        if step == warmup_steps:
            start = time.perf_counter()
            warmup_seconds = start - warmup_start
        environment.step_and_reset(action)
    seconds = time.perf_counter() - start
    return BenchmarkResult(
        backend,
        False,
        transitions,
        seconds,
        seconds,
        0.0,
        warmup_seconds,
        transitions / seconds,
        None,
        None,
    )


def benchmark_configuration(
    *,
    game: str,
    backend: str,
    compile_policy: bool,
    transitions: int,
    warmup_transitions: int,
    num_envs: int,
    batch_size: int,
    seed: int,
    device: str,
    deterministic: bool = True,
    environment_only: bool = False,
    env_threads: int = 4,
    torch_threads: int = 1,
    capture_updates: bool = True,
) -> BenchmarkResult:
    """Warm the real training path, then time complete collect-update cycles."""
    if min(transitions, warmup_transitions, num_envs, batch_size, env_threads, torch_threads) <= 0:
        raise ValueError("transition budgets, environment counts, and batch size must be positive")
    if transitions % num_envs != 0 or warmup_transitions % num_envs != 0:
        raise ValueError("transition budgets must be divisible by num_envs")

    configure_ppo_runtime(backend)
    seed_everything(seed, deterministic=deterministic)
    environment = make_vector_atari_env(
        game, num_envs, backend=backend, seed=seed, num_threads=env_threads
    )
    torch.set_num_threads(torch_threads)
    if environment_only:
        try:
            return benchmark_environment(
                environment,
                transitions=transitions,
                warmup_transitions=warmup_transitions,
                seed=seed,
                backend=backend,
            )
        finally:
            environment.close()
    try:
        agent = PPOAgent(state_dim=4, action_dim=environment.action_space, device=device)
        learner = PPOLearner(
            agent,
            update_epochs=4,
            minibatch_size=batch_size,
            compile_policy=compile_policy,
            capture_updates=capture_updates,
        )
        collector = PPOCollector(environment, learner, rollout_length=128)
        warmup_start = time.perf_counter()
        warmup_remaining = warmup_transitions
        while warmup_remaining:
            rollout = collector.collect(warmup_remaining)
            learner.update(rollout)
            warmup_remaining -= rollout.transition_count

        if agent.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        warmup_seconds = time.perf_counter() - warmup_start

        start = time.perf_counter()
        collection_seconds = 0.0
        update_seconds = 0.0
        remaining = transitions
        while remaining:
            phase_start = time.perf_counter()
            rollout = collector.collect(remaining)
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            collection_seconds += time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            learner.update(rollout)
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            update_seconds += time.perf_counter() - phase_start
            remaining -= rollout.transition_count
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        if agent.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated() / 2**20
            peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        else:
            peak_allocated = None
            peak_reserved = None
        return BenchmarkResult(
            backend=backend,
            compiled=compile_policy,
            transitions=transitions,
            seconds=seconds,
            collection_seconds=collection_seconds,
            update_seconds=update_seconds,
            warmup_seconds=warmup_seconds,
            transitions_per_second=transitions / seconds,
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
        )
    finally:
        environment.close()


def main() -> None:
    """Run the eager baseline and/or optimized configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pong-v5")
    parser.add_argument("--transitions", type=int, default=10_240)
    parser.add_argument("--warmup-transitions", type=int, default=1_024)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--env-threads", type=int, default=4, help="Native ALE worker threads")
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--capture-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--configuration",
        choices=("baseline", "optimized", "ale", "both", "all"),
        default="both",
    )
    args = parser.parse_args()

    if min(args.batch_size, args.num_envs, args.torch_threads, args.env_threads) <= 0:
        parser.error("batch-size, num-envs, torch-threads, and env-threads must be positive")
    if not args.environment_only and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a CPU benchmark")

    configurations = []
    if args.configuration in {"baseline", "both", "all"}:
        configurations.append(("sync", False))
    if args.configuration in {"optimized", "both", "all"}:
        configurations.append(("async", True))
    if args.configuration in {"ale", "all"}:
        configurations.append(("ale", True))

    results = []
    for backend, compile_policy in configurations:
        result = benchmark_configuration(
            game=args.game,
            backend=backend,
            compile_policy=compile_policy,
            transitions=args.transitions,
            warmup_transitions=args.warmup_transitions,
            num_envs=args.num_envs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            deterministic=args.deterministic,
            environment_only=args.environment_only,
            env_threads=args.env_threads,
            torch_threads=args.torch_threads,
            capture_updates=args.capture_updates,
        )
        results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "game": args.game,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "measurement": "environment" if args.environment_only else "collect_and_update",
        "capture_updates": args.capture_updates,
        "native_ale_note": "Different preprocessing/reset protocol; throughput is not learning speed.",
        "num_envs": args.num_envs,
        "batch_size": args.batch_size,
        "rollout_length": 128,
        "update_epochs": 4,
        "warmup_transitions": args.warmup_transitions,
        "timed_transitions": args.transitions,
        "torch_version": torch.__version__,
        "gymnasium_version": version("gymnasium"),
        "ale_version": version("ale-py"),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name()
        if args.device == "cuda" and not args.environment_only
        else "cpu",
        "cpu": platform.processor(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "torch_threads": torch.get_num_threads(),
        "env_threads": args.env_threads,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,
        "results": [asdict(result) for result in results],
    }
    if len(results) == 2:
        output["speedup"] = results[1].transitions_per_second / results[0].transitions_per_second
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
