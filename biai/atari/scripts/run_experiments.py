"""Run the repository's standard training and evaluation matrices."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from biai.atari.training import DEFAULT_MAX_EPISODE_STEPS
from biai.atari.training.results import (
    case_name,
    check_output,
    file_digest,
    publish_output,
    run_metadata,
)
from biai.paths import PROJECT_ROOT

DEFAULT_GAMES = {
    "single": ("Pong-v5", "Breakout-v5"),
    "continual": ("Pong-v5", "Breakout-v5", "SpaceInvaders-v5"),
    "multitask": ("Pong-v5", "Breakout-v5", "SpaceInvaders-v5"),
}
DEFAULT_TRAINING_STEPS = {
    "single": 500_000,
    "continual": 500_000,
    "multitask": 50_000,
}


@dataclass(frozen=True)
class Job:
    """One child Python process in an experiment matrix."""

    name: str
    arguments: tuple[str, ...]
    log_name: str
    capture_output: bool
    depends_on: tuple[str, ...] = ()
    case_id: str = ""

    @property
    def command(self) -> tuple[str, ...]:
        return (sys.executable, *self.arguments)


def _game_slug(game: str) -> str:
    return game.removesuffix("-v5").replace("/", "_").lower()


def _ewc_variants(mode: str) -> tuple[bool, ...]:
    if mode == "both":
        return (False, True)
    return (mode == "on",)


def build_jobs(
    phase: str,
    suite: str,
    *,
    games: Sequence[str],
    algorithms: Sequence[str],
    steps: int | None = None,
    episodes: int = 10,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ewc_lambda: float = 0.4,
    ewc_mode: str = "both",
    seed: int = 0,
    num_envs: int = 1,
    env_backend: str = "sync",
    env_threads: int = 4,
    compile_ppo: bool = False,
    compile_dqn: bool = False,
    deterministic: bool = False,
    method: str | None = None,
    task_steps: Sequence[int] | None = None,
) -> list[Job]:
    """Build the process matrix formerly encoded in the shell scripts."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if env_backend not in {"sync", "async", "ale"}:
        raise ValueError("env_backend must be sync, async, or ale")
    if method is not None and (suite != "continual" or method not in {"finetune", "ewc", "gpm"}):
        raise ValueError("method selects finetune, ewc, or gpm in the continual suite")
    if method is not None and ewc_mode != "both":
        raise ValueError("Use either method or ewc_mode")
    if task_steps is not None:
        if phase != "train" or suite != "continual":
            raise ValueError("task_steps is supported only for continual training")
        if len(task_steps) != len(games) or any(value <= 0 for value in task_steps):
            raise ValueError("task_steps must provide one positive transition budget per game")
        if steps is not None:
            raise ValueError("Use either steps or task_steps")
    methods = (
        (method,)
        if method
        else tuple("ewc" if flag else "finetune" for flag in _ewc_variants(ewc_mode))
    )
    if phase != "train" and (num_envs != 1 or env_backend != "sync"):
        raise ValueError("Evaluation reads its observation protocol from the checkpoint")
    if env_threads <= 0:
        raise ValueError("env_threads must be positive")
    if compile_ppo and tuple(algorithms) != ("ppo",):
        raise ValueError("compile_ppo supports PPO only")
    if compile_dqn and tuple(algorithms) != ("dqn",):
        raise ValueError("compile_dqn supports DQN only")
    if phase == "train":
        training_steps = steps if steps is not None else DEFAULT_TRAINING_STEPS[suite]
        if training_steps <= 0:
            raise ValueError("steps must be positive")
        budgets = task_steps if task_steps is not None else (training_steps,)
        if num_envs > 1:
            if any(value % num_envs for value in budgets):
                raise ValueError("Task budgets must be divisible by num_envs")
        jobs = _build_training_jobs(
            suite,
            games,
            algorithms,
            training_steps,
            episodes,
            max_steps,
            ewc_lambda,
            methods,
            seed,
            num_envs,
            env_backend,
            compile_ppo,
            task_steps,
        )
        if env_threads != 4:
            jobs = [
                replace(job, arguments=(*job.arguments, "--env-threads", str(env_threads)))
                for job in jobs
            ]
        if compile_dqn:
            jobs = [replace(job, arguments=(*job.arguments, "--compile-dqn")) for job in jobs]
        if deterministic:
            jobs = [replace(job, arguments=(*job.arguments, "--deterministic")) for job in jobs]
        return jobs

    jobs = _build_evaluation_jobs(
        suite, games, algorithms, episodes, max_steps, ewc_lambda, methods, seed
    )
    if compile_dqn:
        jobs = [replace(job, arguments=(*job.arguments, "--compile-dqn")) for job in jobs]
    if compile_ppo:
        jobs = [replace(job, arguments=(*job.arguments, "--compile-ppo")) for job in jobs]
    if deterministic:
        jobs = [replace(job, arguments=(*job.arguments, "--deterministic")) for job in jobs]
    return jobs


def _build_training_jobs(
    suite: str,
    games: Sequence[str],
    algorithms: Sequence[str],
    steps: int,
    eval_episodes: int,
    eval_max_steps: int,
    ewc_lambda: float,
    methods: Sequence[str],
    seed: int,
    num_envs: int,
    env_backend: str,
    compile_ppo: bool,
    task_steps: Sequence[int] | None,
) -> list[Job]:
    jobs = []
    if suite == "single":
        for game in games:
            for algorithm in algorithms:
                case = case_name("single", algorithm, game=game, seed=seed)
                name = f"single {game} {algorithm}"
                arguments = [
                    "-m",
                    "biai.atari.scripts.train_single",
                    "--output-dir",
                    case,
                    "--seed",
                    str(seed),
                    "--game",
                    game,
                    "--eval-episodes",
                    str(eval_episodes),
                    "--algorithm",
                    algorithm,
                    "--steps",
                    str(steps),
                ]
                if num_envs > 1:
                    arguments.extend(("--num-envs", str(num_envs)))
                if env_backend != "sync":
                    arguments.extend(("--env-backend", env_backend))
                if algorithm == "ppo" and compile_ppo:
                    arguments.append("--compile-ppo")
                jobs.append(
                    Job(
                        name,
                        tuple(arguments),
                        f"{_game_slug(game)}_{algorithm}_seed{seed}.log",
                        True,
                        case_id=case,
                    )
                )
    elif suite == "continual":
        for algorithm in algorithms:
            for method in methods:
                case = case_name("continual", algorithm, method=method, seed=seed)
                arguments = [
                    "-m",
                    "biai.atari.scripts.train_continual",
                    "--output-dir",
                    case,
                    "--seed",
                    str(seed),
                    "--games",
                    *games,
                    "--algorithm",
                    algorithm,
                    "--steps-per-game",
                    str(steps),
                    "--eval-episodes",
                    str(eval_episodes),
                    "--eval-max-steps",
                    str(eval_max_steps),
                ]
                variant = "no_ewc" if method == "finetune" else method
                if method == "ewc":
                    arguments.extend(("--use-ewc", "--ewc-lambda", str(ewc_lambda)))
                elif method == "gpm":
                    arguments.extend(("--method", "gpm"))
                if task_steps is not None:
                    arguments.extend(("--task-steps", *(str(value) for value in task_steps)))
                if num_envs > 1:
                    arguments.extend(("--num-envs", str(num_envs)))
                if env_backend != "sync":
                    arguments.extend(("--env-backend", env_backend))
                if algorithm == "ppo" and compile_ppo:
                    arguments.append("--compile-ppo")
                jobs.append(
                    Job(
                        f"continual {algorithm} {variant}",
                        tuple(arguments),
                        f"continual_{algorithm}_{variant}_seed{seed}.log",
                        True,
                        case_id=case,
                    )
                )
    else:
        for algorithm in algorithms:
            case = case_name("multitask", algorithm, seed=seed)
            jobs.append(
                Job(
                    f"multitask {algorithm}",
                    (
                        "-m",
                        "biai.atari.scripts.train_multitask",
                        "--output-dir",
                        case,
                        "--seed",
                        str(seed),
                        "--games",
                        *games,
                        "--algorithm",
                        algorithm,
                        "--steps",
                        str(steps),
                        *(("--compile-ppo",) if compile_ppo else ()),
                        *(("--num-envs", str(num_envs)) if num_envs != 1 else ()),
                        *(("--env-backend", env_backend) if env_backend != "sync" else ()),
                    ),
                    f"multitask_{algorithm}_seed{seed}.log",
                    True,
                    case_id=case,
                )
            )
    return jobs


def _build_evaluation_jobs(
    suite: str,
    games: Sequence[str],
    algorithms: Sequence[str],
    episodes: int,
    max_steps: int,
    ewc_lambda: float,
    methods: Sequence[str],
    seed: int,
) -> list[Job]:
    jobs = []
    if suite == "single":
        for game in games:
            for algorithm in algorithms:
                case = case_name("single", algorithm, game=game, seed=seed)
                jobs.append(
                    Job(
                        f"evaluate single {game} {algorithm}",
                        (
                            "-m",
                            "biai.atari.scripts.evaluate",
                            "--seed",
                            str(seed),
                            "--mode",
                            "single",
                            "--model",
                            f"{case}/checkpoints/final.pt",
                            "--algorithm",
                            algorithm,
                            "--game",
                            game,
                            "--episodes",
                            str(episodes),
                            "--max-steps",
                            str(max_steps),
                            "--json-out",
                            f"{case}/evaluation/evaluation.json",
                        ),
                        f"evaluate_{_game_slug(game)}_{algorithm}_seed{seed}.log",
                        False,
                        case_id=case,
                    )
                )
    elif suite == "continual":
        for algorithm in algorithms:
            for method in methods:
                case = case_name("continual", algorithm, method=method, seed=seed)
                use_ewc = method == "ewc"
                variant = f"{algorithm}_gpm" if method == "gpm" else f"{algorithm}_ewc{use_ewc}"
                arguments = [
                    "-m",
                    "biai.atari.scripts.evaluate",
                    "--seed",
                    str(seed),
                    "--mode",
                    "continual",
                    "--model",
                    f"{case}/checkpoints/final.pt",
                    "--algorithm",
                    algorithm,
                    "--games",
                    *games,
                    "--episodes",
                    str(episodes),
                    "--max-steps",
                    str(max_steps),
                    "--json-out",
                    f"{case}/evaluation/evaluation.json",
                ]
                if use_ewc:
                    arguments.extend(("--ewc", "--ewc-lambda", str(ewc_lambda)))
                jobs.append(
                    Job(
                        f"evaluate continual {algorithm} {method}",
                        tuple(arguments),
                        f"evaluate_continual_{variant}_seed{seed}.log",
                        False,
                        case_id=case,
                    )
                )
    else:
        for algorithm in algorithms:
            case = case_name("multitask", algorithm, seed=seed)
            jobs.append(
                Job(
                    f"evaluate multitask {algorithm}",
                    (
                        "-m",
                        "biai.atari.scripts.evaluate",
                        "--seed",
                        str(seed),
                        "--mode",
                        "multitask",
                        "--model",
                        f"{case}/checkpoints/final.pt",
                        "--algorithm",
                        algorithm,
                        "--games",
                        *games,
                        "--episodes",
                        str(episodes),
                        "--max-steps",
                        str(max_steps),
                        "--json-out",
                        f"{case}/evaluation/evaluation.json",
                    ),
                    f"evaluate_multitask_{algorithm}_seed{seed}.log",
                    False,
                    case_id=case,
                )
            )
    return jobs


def build_teaching_jobs(
    *,
    phase: str,
    seed: int,
    steps: int | None = None,
    episodes: int = 10,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ewc_lambda: float = 0.4,
    num_envs: int = 8,
    env_backend: str = "async",
    dqn_num_envs: int = 8,
    env_threads: int = 4,
    compile_ppo: bool = True,
    compile_dqn: bool = True,
    deterministic: bool = True,
    smoke: bool = False,
    matched_budget: bool = False,
) -> list[Job]:
    """Build the reference or matched matrix, including a final evaluation of each job.

    matched_budget adds the third single-task baseline and exact joint-game quotas.
    The default single-task suite uses two teaching games. Joint training gets the same total
    transition budget as the original three-task protocol. Without a uniform steps
    override, Pong single-task uses 2M and continual PPO uses the verified budgets.
    """
    if smoke and steps is not None:
        raise ValueError("smoke defines its own algorithm-specific budgets; omit steps")
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive")
    if smoke:
        episodes = 1
    training, evaluation = [], []
    for algorithm in ("ppo", "dqn"):
        for suite, method in (
            ("single", None),
            ("multitask", None),
            ("continual", "finetune"),
            ("continual", "ewc"),
            ("continual", "gpm"),
        ):
            options = dict(
                games=DEFAULT_GAMES["multitask"] if matched_budget else DEFAULT_GAMES[suite],
                algorithms=(algorithm,),
                seed=seed,
                episodes=episodes,
                max_steps=max_steps,
                method=method,
                ewc_lambda=ewc_lambda,
                compile_dqn=compile_dqn and algorithm == "dqn",
                compile_ppo=compile_ppo and algorithm == "ppo",
            )
            if phase == "train":
                case_envs = num_envs if algorithm == "ppo" else dqn_num_envs
                case_backend = (
                    env_backend
                    if algorithm == "ppo" or dqn_num_envs > 1 or env_backend == "ale"
                    else "sync"
                )
                budgets = (
                    (1_000_448, 500_000, 500_000)
                    if (
                        steps is None
                        and not smoke
                        and not matched_budget
                        and suite == "continual"
                        and algorithm == "ppo"
                    )
                    else None
                )
                case_steps = (
                    (10_032 if algorithm == "dqn" else 2_048) if smoke else (steps or 500_000)
                )
                if suite == "multitask":
                    case_steps *= len(DEFAULT_GAMES[suite])
                suite_jobs = build_jobs(
                    "train",
                    suite,
                    **options,
                    steps=None if budgets else case_steps,
                    task_steps=budgets,
                    num_envs=case_envs,
                    env_backend=case_backend,
                    env_threads=env_threads,
                    deterministic=deterministic,
                )
                if matched_budget and suite == "multitask":
                    quota = case_steps // len(options["games"])
                    if quota % case_envs:
                        raise ValueError("Each matched game budget must be divisible by num_envs")
                    suite_jobs = [
                        replace(job, arguments=(*job.arguments, "--steps-per-game", str(quota)))
                        for job in suite_jobs
                    ]
                training.extend(suite_jobs)
            evaluation.extend(build_jobs("evaluate", suite, **options, deterministic=deterministic))
    if smoke:
        training = [
            replace(
                job,
                arguments=(*job.arguments, "--gpm-samples", "16", "--gpm-collection-steps", "32"),
            )
            if "--method" in job.arguments and "gpm" in job.arguments
            else job
            for job in training
        ]
    if steps is None and not smoke:
        updated = []
        for job in training:
            arguments = list(job.arguments)
            if (
                not matched_budget
                and arguments[1] == "biai.atari.scripts.train_single"
                and "Pong-v5" in arguments
            ):
                arguments[arguments.index("--steps") + 1] = "2000000"
            if arguments[1] in {
                "biai.atari.scripts.train_single",
                "biai.atari.scripts.train_continual",
            }:
                arguments.extend(("--eval-points", "10"))
            updated.append(replace(job, arguments=tuple(arguments)))
        training = updated
    if training:
        evaluation = [
            replace(evaluate, depends_on=(train.log_name,))
            for train, evaluate in zip(training, evaluation, strict=True)
        ]
    return [replace(job, capture_output=True) for job in training + evaluation]


def _cuda_device_count(device: str) -> int:
    if device == "cpu":
        return 0

    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device == "cuda" and count == 0:
        raise RuntimeError("--device cuda was requested, but PyTorch found no CUDA devices")
    return count


def _job_environment(device_count: int, job_index: int, parallel: bool) -> dict[str, str]:
    environment = os.environ.copy()
    if device_count == 0:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(job_index % device_count if parallel else 0)
    return environment


def _print_job(job: Job, index: int, total: int, device: str) -> None:
    print(f"[{index}/{total}] {job.name} ({device})")
    print(f"  {shlex.join(job.command)}")


def run_jobs(
    jobs: Sequence[Job],
    *,
    device: str,
    parallel: bool,
    log_dir: Path,
    max_workers: int | None = None,
    cpus_per_job: int | None = None,
    run_dir: Path | None = None,
) -> None:
    """Bound concurrency, respect checkpoint dependencies, and record every child outcome."""
    workers = max_workers if max_workers is not None else (2 if parallel else 1)
    if workers <= 0 or (cpus_per_job is not None and cpus_per_job <= 0):
        raise ValueError("Worker and CPU counts must be positive")
    identifiers = {job.log_name for job in jobs}
    if len(identifiers) != len(jobs):
        raise ValueError("Job log names must be unique")
    if any(dependency not in identifiers for job in jobs for dependency in job.depends_on):
        raise ValueError("Job dependency is missing from this matrix")
    device_count = _cuda_device_count(device)
    cwd = PROJECT_ROOT if run_dir is None else run_dir.resolve()
    if run_dir is not None:
        if cwd.exists():
            raise ValueError(f"Training requires a new run directory: {cwd}")
        cwd.mkdir(parents=True)
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    available_cpus = sorted(os.sched_getaffinity(0))
    cpu_count = cpus_per_job or (
        max(1, len(available_cpus) // workers) if workers > 1 else len(available_cpus)
    )
    if cpu_count * workers > len(available_cpus):
        raise ValueError("Requested CPU allocation exceeds the available affinity set")
    status_path = (cwd if run_dir is not None else log_dir) / "run.json"
    if status_path.exists():
        raise FileExistsError(
            f"Run record already exists; choose a new log directory: {status_path}"
        )
    status = {
        **run_metadata(PROJECT_ROOT),
        "cases": sorted({job.case_id for job in jobs if job.case_id}),
        "pid": os.getpid(),
        "max_workers": workers,
        "jobs": [
            {
                "name": job.name,
                "command": list(job.command),
                "case_id": job.case_id,
                "log": os.path.relpath(
                    log_dir
                    / (
                        Path(job.case_id)
                        / ("train.log" if "--output-dir" in job.arguments else "evaluate.log")
                        if job.case_id
                        else job.log_name
                    ),
                    status_path.parent,
                ),
                "depends_on": list(job.depends_on),
                "state": "pending",
            }
            for job in jobs
        ],
    }

    def write_status():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    pending, running, completed = list(enumerate(jobs)), {}, set()
    write_status()
    try:
        while pending or running:
            for slot, (index, job, process, log_file) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                if log_file is not None:
                    log_file.close()
                del running[slot]
                status["jobs"][index].update(
                    state="completed" if code == 0 else "failed",
                    exit_code=code,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                write_status()
                if code:
                    raise RuntimeError(f"Job failed with exit code {code}: {job.name}")
                completed.add(job.log_name)
            for slot in range(workers):
                if slot in running:
                    continue
                ready = next(
                    ((i, job) for i, job in pending if set(job.depends_on) <= completed), None
                )
                if ready is None:
                    continue
                index, job = ready
                pending.remove(ready)
                environment = _job_environment(device_count, slot, parallel=workers > 1)
                environment.update(
                    PYTHONPATH=str(PROJECT_ROOT),
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                cpus = available_cpus[slot * cpu_count : (slot + 1) * cpu_count]
                command = ("taskset", "-c", ",".join(map(str, cpus)), *job.command)
                log_path = status_path.parent / status["jobs"][index]["log"]
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("w") if job.capture_output or workers > 1 else None
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT if log_file is not None else None,
                        start_new_session=True,
                    )
                except BaseException:
                    if log_file is not None:
                        log_file.close()
                    raise
                running[slot] = (index, job, process, log_file)
                status["jobs"][index].update(
                    state="running",
                    pid=process.pid,
                    cpus=cpus,
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
                _print_job(
                    job,
                    index + 1,
                    len(jobs),
                    "cpu" if not device_count else f"cuda:{slot % device_count}",
                )
                write_status()
            if pending and not running:
                raise ValueError("Job dependencies contain a cycle")
            if running:
                time.sleep(0.1)
    finally:
        for index, _, process, log_file in running.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if log_file is not None:
                log_file.close()
            status["jobs"][index].update(state="cancelled", exit_code=process.returncode)
        write_status()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("suite", choices=("single", "continual", "multitask", "teaching"))
    parser.add_argument(
        "--matched-budget",
        action="store_true",
        help="Teaching: all three single baselines and identical per-game budgets",
    )
    parser.add_argument("--games", nargs="+", help="Override the suite's default games")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("dqn", "ppo"),
        default=("dqn", "ppo"),
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--steps", type=int, help="Training steps or steps per game")
    parser.add_argument("--task-steps", type=int, nargs="+", help="One budget per sequential task")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes per game")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Maximum agent steps per episode; incomplete episodes are rejected",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=0)
    seed_group.add_argument("--seeds", type=int, nargs="+", help="Expand the matrix over seeds")
    parser.add_argument("--ewc-lambda", type=float, default=0.4)
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument(
        "--method", choices=("finetune", "ewc", "gpm"), help="Continual-learning method"
    )
    methods.add_argument(
        "--ewc-mode",
        choices=("both", "on", "off"),
        default="both",
        help="Continual suite variants to run",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Environments per game (teaching PPO: 8; teaching DQN uses --dqn-num-envs; other suites: 1)",
    )
    parser.add_argument(
        "--env-backend",
        choices=("sync", "async", "ale"),
        default=None,
        help="Environment backend; ALE is a distinct preprocessing protocol",
    )
    parser.add_argument(
        "--compile-ppo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile PPO rollout and learner policy evaluation (enabled for teaching)",
    )
    parser.add_argument(
        "--compile-dqn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile DQN inference and TD loss (enabled for teaching)",
    )
    parser.add_argument(
        "--dqn-num-envs",
        type=int,
        default=8,
        help="DQN environments per game in teaching (default: 8)",
    )
    parser.add_argument("--env-threads", type=int, default=4, help="Native ALE threads per game")
    parser.add_argument("--parallel", action="store_true", help="Run at most two jobs concurrently")
    parser.add_argument("--max-workers", type=int, help="Maximum simultaneous jobs (default: 1)")
    parser.add_argument(
        "--cpus-per-job", type=int, help="Disjoint CPU cores assigned to each worker"
    )
    parser.add_argument(
        "--results-dir", type=Path, help="Case output root (default: results, or results/smoke)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace selected existing results after success"
    )
    parser.add_argument(
        "--eval-points", type=int, help="Process evaluation points per single/continual task budget"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short teaching runs covering learning and all boundaries",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return parser


def _dispatch_jobs(jobs, args):
    if args.eval_points is not None:
        if args.eval_points < 0 or args.phase != "train" or args.suite == "multitask":
            raise ValueError(
                "eval-points requires single, continual or teaching training and a nonnegative count"
            )
        updated = []
        for job in jobs:
            arguments = list(job.arguments)
            if arguments[1] in {
                "biai.atari.scripts.train_single",
                "biai.atari.scripts.train_continual",
            }:
                if "--eval-points" in arguments:
                    arguments[arguments.index("--eval-points") + 1] = str(args.eval_points)
                else:
                    arguments.extend(("--eval-points", str(args.eval_points)))
            updated.append(replace(job, arguments=tuple(arguments)))
        jobs = updated
    default_folder = "smoke" if args.smoke else ""
    if args.matched_budget:
        default_folder = "atari-matched-smoke" if args.smoke else "atari-matched"
    root = (args.results_dir or PROJECT_ROOT / "results" / default_folder).resolve()
    cases = sorted({job.case_id for job in jobs})
    destinations = {
        case: root / case / ("evaluation" if args.phase == "evaluate" else "") for case in cases
    }
    if args.phase == "evaluate":
        updated = []
        for job in jobs:
            arguments = list(job.arguments)
            model_index = arguments.index("--model") + 1
            arguments[model_index] = str(root / arguments[model_index])
            updated.append(replace(job, arguments=tuple(arguments)))
        jobs = updated
    if args.dry_run:
        print(f"Results directory: {root}")
        for index, job in enumerate(jobs, start=1):
            _print_job(job, index, len(jobs), args.device)
        return
    for destination in destinations.values():
        check_output(destination, force=args.force)
    if args.phase == "evaluate":
        for job in jobs:
            model = Path(job.arguments[job.arguments.index("--model") + 1])
            if not model.is_file():
                raise FileNotFoundError(model)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    work = staging / "work"
    jobs = [replace(job, capture_output=True) for job in jobs]
    try:
        run_jobs(
            jobs,
            device=args.device,
            parallel=args.parallel,
            log_dir=work / ".logs",
            max_workers=args.max_workers,
            cpus_per_job=args.cpus_per_job,
            run_dir=work,
        )
        record = json.loads((work / "run.json").read_text())
        # Check every selected result before replacing any existing case.
        for job in jobs:
            directory = work / job.case_id
            if "--output-dir" in job.arguments:
                summary = json.loads((directory / "training_summary.json").read_text())
                json.loads((directory / "config.json").read_text())
                if file_digest(directory / "checkpoints/final.pt") != summary["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {job.case_id}")
            else:
                json.loads((directory / "evaluation/evaluation.json").read_text())
        for job, row in zip(jobs, record["jobs"], strict=True):
            training = "--output-dir" in job.arguments
            directory = work / job.case_id / ("" if training else "evaluation")
            metadata_path = directory / "run.json"
            metadata = (
                json.loads(metadata_path.read_text())
                if metadata_path.exists()
                else {k: record[k] for k in ("created_at", "commit", "dirty", "versions")}
            )
            log_name = "train.log" if training else "evaluate.log"
            shutil.move(work / row["log"], directory / log_name)
            metadata.update(case=job.case_id, jobs=[{**row, "log": log_name}])
            if not training:
                checkpoint = (
                    (root if args.phase == "evaluate" else work)
                    / job.case_id
                    / "checkpoints/final.pt"
                )
                metadata.update(
                    checkpoint="../checkpoints/final.pt", checkpoint_sha256=file_digest(checkpoint)
                )
            if args.smoke and args.phase == "train":
                metadata["temporary_models_and_videos_removed"] = True
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if args.smoke and args.phase == "train":
            for pattern in ("*/checkpoints", "*/videos", "*/evaluation/videos"):
                for directory in work.glob(pattern):
                    shutil.rmtree(directory)
        for case, destination in destinations.items():
            source = work / case / ("evaluation" if args.phase == "evaluate" else "")
            publish_output(source, destination, force=args.force)
        shutil.rmtree(staging)
    except BaseException:
        print(f"Unpublished results and logs retained at {work}", file=sys.stderr)
        raise
    print(f"Completed results saved to {root}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the experiment matrix and launch its Python child processes."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = args.seeds or (args.seed,)
        if args.suite == "teaching":
            if args.games or args.method or args.task_steps or args.ewc_mode != "both":
                raise ValueError("teaching uses fixed games and methods")
            if tuple(args.algorithms) != ("dqn", "ppo"):
                raise ValueError("teaching includes both DQN and PPO")
            jobs = []
            for seed in seeds:
                jobs.extend(
                    build_teaching_jobs(
                        phase=args.phase,
                        smoke=args.smoke,
                        matched_budget=args.matched_budget,
                        seed=seed,
                        steps=args.steps,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                        ewc_lambda=args.ewc_lambda,
                        num_envs=8 if args.num_envs is None else args.num_envs,
                        env_backend=args.env_backend or "async",
                        dqn_num_envs=args.dqn_num_envs,
                        env_threads=args.env_threads,
                        compile_ppo=True if args.compile_ppo is None else args.compile_ppo,
                        compile_dqn=True if args.compile_dqn is None else args.compile_dqn,
                        deterministic=True if args.deterministic is None else args.deterministic,
                    )
                )
            _dispatch_jobs(jobs, args)
            return
        if args.matched_budget:
            raise ValueError("matched-budget is defined for the teaching suite")
        if args.smoke:
            raise ValueError("smoke is defined for the teaching suite")
        games = args.games or DEFAULT_GAMES[args.suite]
        jobs = []
        for seed in seeds:
            jobs.extend(
                build_jobs(
                    args.phase,
                    args.suite,
                    games=games,
                    algorithms=args.algorithms,
                    steps=args.steps,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    ewc_lambda=args.ewc_lambda,
                    ewc_mode=args.ewc_mode,
                    seed=seed,
                    num_envs=1 if args.num_envs is None else args.num_envs,
                    env_backend=args.env_backend or "sync",
                    env_threads=args.env_threads,
                    compile_ppo=bool(args.compile_ppo),
                    compile_dqn=bool(args.compile_dqn),
                    deterministic=args.deterministic,
                    method=args.method,
                    task_steps=args.task_steps,
                )
            )
        _dispatch_jobs(jobs, args)
    except (RuntimeError, ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
