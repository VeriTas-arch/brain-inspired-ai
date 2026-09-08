"""Benchmark the maintained PPO runtime without changing its minibatch size."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass

import torch

from algorithms import PPOAgent
from environments import make_vector_atari_env
from training import PPOCollector, PPOLearner, seed_everything


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing and PyTorch allocator measurements for one runtime configuration."""

    backend: str
    compiled: bool
    transitions: int
    seconds: float
    transitions_per_second: float
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None


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
) -> BenchmarkResult:
    """Warm the real training path, then time complete collect-update cycles."""
    if transitions <= 0 or warmup_transitions <= 0:
        raise ValueError("transition budgets must be positive")
    if transitions % num_envs != 0 or warmup_transitions % num_envs != 0:
        raise ValueError("transition budgets must be divisible by num_envs")

    seed_everything(seed)
    environment = make_vector_atari_env(
        game,
        num_envs,
        backend=backend,
        seed=seed,
    )
    agent = PPOAgent(state_dim=4, action_dim=environment.action_space, device=device)
    learner = PPOLearner(
        agent,
        update_epochs=4,
        minibatch_size=batch_size,
        compile_policy=compile_policy,
    )
    collector = PPOCollector(environment, learner, rollout_length=128)

    try:
        warmup_remaining = warmup_transitions
        while warmup_remaining:
            rollout = collector.collect(warmup_remaining)
            learner.update(rollout)
            warmup_remaining -= rollout.transition_count

        if agent.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        remaining = transitions
        while remaining:
            rollout = collector.collect(remaining)
            learner.update(rollout)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--configuration",
        choices=("baseline", "optimized", "both"),
        default="both",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_envs <= 0:
        parser.error("batch-size and num-envs must be positive")

    torch.set_num_threads(1)
    configurations = []
    if args.configuration in {"baseline", "both"}:
        configurations.append(("sync", False))
    if args.configuration in {"optimized", "both"}:
        configurations.append(("async", True))

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
        )
        results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {"batch_size": args.batch_size, "results": [asdict(result) for result in results]}
    if len(results) == 2:
        output["speedup"] = results[1].transitions_per_second / results[0].transitions_per_second
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
