"""Measure DQN inference, updates, and the complete Atari/replay/learning path."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch

from algorithms import DQNAgent, EWCWrapper, MultiHeadDQNAgent
from algorithms.subspace_projection import AdamSubspaceProjection, build_input_subspaces
from environments import AtariEnv, make_vector_atari_env, observation_protocol
from training import (
    DQNMetrics,
    ReplayBuffer,
    configure_ppo_runtime,
    dqn_updates_due,
    seed_everything,
)


def benchmark_configuration(
    *,
    component="pipeline",
    variant="single",
    compiled=False,
    game="Pong-v5",
    transitions=8192,
    warmup=1024,
    batch_size=32,
    replay_size=10000,
    epsilon=0.1,
    seed=0,
    device="cuda",
    num_envs=1,
    env_backend="sync",
    env_threads=4,
    capture_updates=True,
    defer_metrics=True,
):
    """Use fixed epsilon, batch size, and one update per four collected transitions.

    EWC/GPM measurements exercise a second task after a real consolidation using 16
    observations. They measure computation, not old-task retention or convergence.
    Replay prefill and compiler warmup are excluded from steady-state throughput.
    """
    if min(transitions, warmup, batch_size, replay_size) <= 0 or replay_size < batch_size:
        raise ValueError("Positive budgets and replay_size >= batch_size are required")
    if component == "pipeline" and (transitions % 4 or warmup % 4):
        raise ValueError("Pipeline budgets must be divisible by four")
    if num_envs <= 0 or any(value % num_envs for value in (transitions, warmup, replay_size)):
        raise ValueError("Budgets and replay size must be divisible by positive num_envs")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")
    configure_ppo_runtime(env_backend)
    torch.set_num_threads(1)
    seed_everything(seed, deterministic=True)
    vectorized = num_envs > 1 or env_backend != "sync"
    env = (
        make_vector_atari_env(
            game, num_envs, backend=env_backend, seed=seed, num_threads=env_threads
        )
        if vectorized
        else AtariEnv(game, seed=seed)
    )
    projection = None
    try:
        if variant == "single":
            base = DQNAgent(
                4, env.action_space, epsilon_start=epsilon, epsilon_end=epsilon, device=device
            )
        else:
            base = MultiHeadDQNAgent(4, epsilon=epsilon, epsilon_decay=1.0, device=device)
            base.register_task("old", env.action_space)
            base.set_task("old")
        agent = EWCWrapper(base) if variant == "ewc" else base
        replay = ReplayBuffer(capacity=max(100000, replay_size), seed=(seed, 0, 1))
        state = env.reset()
        prefill_start = time.perf_counter()
        for index in range(0, replay_size, num_envs):
            # Deterministic actions keep policy/replay RNG streams untouched during prefill.
            if vectorized:
                before = state.clone()
                actions = torch.arange(index, index + num_envs) % env.action_space
                transition = env.step_and_reset(actions)
                replay.add_batch(
                    before,
                    actions,
                    transition.rewards,
                    transition.transition_observations,
                    transition.terminated,
                )
                state = transition.observations
            else:
                action = index % env.action_space
                next_state, reward, terminated, truncated = env.step(action)
                replay.add(state, action, reward, next_state, terminated)
                state = env.reset() if terminated or truncated else next_state
        prefill_seconds = time.perf_counter() - prefill_start
        batch = replay.sample(batch_size)
        if variant in {"ewc", "gpm"}:
            base.update(batch)  # Inherit nonzero Adam moments across the boundary.
            boundary = {name: value[:16] for name, value in batch.items()}
            if variant == "ewc":
                agent.consolidate_weights(boundary)
            else:
                subspaces = build_input_subspaces(
                    base.backbone.network, boundary["states"], threshold=0.97, seed=seed
                )
            agent.register_task("new", env.action_space)
            agent.set_task("new")
        if hasattr(agent, "configure_runtime"):
            agent.configure_runtime(compile_enabled=compiled, capture_updates=capture_updates)
        elif compiled:
            raise RuntimeError("This source snapshot does not support DQN compilation")
        if variant == "gpm":
            options = {"compile_projection": True} if compiled else {}
            projection = AdamSubspaceProjection(
                base.optimizer, base.backbone.network, subspaces, **options
            )

        pending_metrics = DQNMetrics(agent, lambda metrics: None)

        def run(count):
            nonlocal state
            inference_seconds = environment_seconds = replay_seconds = update_seconds = 0.0
            increment = num_envs if component == "pipeline" else 1
            for step in range(0, count, increment):
                start = time.perf_counter()
                if component in {"pipeline", "inference"}:
                    if vectorized:
                        action = (
                            torch.tensor(
                                [
                                    agent.select_action(
                                        state[0], deterministic=component == "inference"
                                    )
                                ]
                            )
                            if num_envs == 1
                            else agent.select_actions(state, deterministic=component == "inference")
                        )
                    else:
                        action = agent.select_action(state, deterministic=component == "inference")
                    inference_seconds += time.perf_counter() - start
                if component == "pipeline":
                    before = state.clone() if vectorized else state
                    start = time.perf_counter()
                    transition = env.step_and_reset(action) if vectorized else env.step(action)
                    environment_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    if vectorized:
                        replay.add_batch(
                            before,
                            action,
                            transition.rewards,
                            transition.transition_observations,
                            transition.terminated,
                        )
                        state = transition.observations
                    else:
                        next_state, reward, terminated, truncated = transition
                        replay.add(before, action, reward, next_state, terminated)
                        state = next_state
                    replay_seconds += time.perf_counter() - start
                    if not vectorized and (terminated or truncated):
                        start = time.perf_counter()
                        state = env.reset()
                        environment_seconds += time.perf_counter() - start
                updates = (
                    1
                    if component == "update"
                    else (
                        dqn_updates_due(step, increment, learning_starts=0)
                        if component == "pipeline"
                        else 0
                    )
                )
                for _ in range(updates):
                    start = time.perf_counter()
                    update_batch = replay.sample(batch_size) if component == "pipeline" else batch
                    replay_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    agent.update(
                        update_batch, metrics_sink=pending_metrics.record if defer_metrics else None
                    )
                    update_seconds += time.perf_counter() - start
            pending_metrics.flush()
            return dict(
                inference_seconds=inference_seconds,
                environment_seconds=environment_seconds,
                replay_seconds=replay_seconds,
                update_seconds=update_seconds,
            )

        start = time.perf_counter()
        run(warmup)
        if base.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        warmup_seconds = time.perf_counter() - start
        start_updates = base.update_count
        start = time.perf_counter()
        phases = run(transitions)
        if base.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return dict(
            component=component,
            variant=variant,
            compiled=compiled,
            capture_updates=compiled and capture_updates,
            defer_metrics=defer_metrics,
            game=game,
            seed=seed,
            deterministic=True,
            epsilon=epsilon,
            batch_size=batch_size,
            replay_size=replay_size,
            replay_sampler="independent_pcg64_without_replacement",
            replay_seed=[seed, 0, 1],
            replay_storage="contiguous_cpu_arrays",
            num_envs=num_envs,
            env_backend=env_backend,
            env_threads=env_threads,
            environment_protocol=observation_protocol(env_backend),
            warmup_iterations=warmup,
            timed_iterations=transitions,
            timed_transitions=transitions if component == "pipeline" else 0,
            timed_updates=base.update_count - start_updates,
            seconds=seconds,
            iterations_per_second=transitions / seconds,
            warmup_seconds=warmup_seconds,
            prefill_seconds=prefill_seconds,
            **phases,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if base.device.type == "cuda" else None,
            cpu=platform.processor(),
            cpu_model=next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            compile_mode="reduce-overhead" if compiled else None,
            fullgraph=compiled,
            torch_threads=torch.get_num_threads(),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20
            if base.device.type == "cuda"
            else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20
            if base.device.type == "cuda"
            else None,
            projection=projection.metrics() if projection is not None else None,
        )
    finally:
        if projection is not None:
            projection.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component", choices=("pipeline", "inference", "update"), default="pipeline"
    )
    parser.add_argument("--variant", choices=("single", "multi", "ewc", "gpm"), default="single")
    parser.add_argument("--compile-dqn", action="store_true")
    parser.add_argument("--game", default="Pong-v5")
    parser.add_argument(
        "--transitions", type=int, default=8192, help="Timed transitions, updates, or inferences"
    )
    parser.add_argument("--warmup", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-backend", choices=("sync", "async", "ale"), default="sync")
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--capture-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--defer-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    args["compiled"] = args.pop("compile_dqn")
    result = benchmark_configuration(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
