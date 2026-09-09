"""Run the repository's standard training and evaluation matrices."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from training import DEFAULT_MAX_EPISODE_STEPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    compile_ppo: bool = False,
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
    if env_backend not in {"sync", "async"}:
        raise ValueError("env_backend must be 'sync' or 'async'")
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
    if num_envs > 1 and (phase != "train" or suite == "multitask" or tuple(algorithms) != ("ppo",)):
        raise ValueError("num_envs greater than one supports PPO single/continual training only")
    if env_backend != "sync" and (
        phase != "train" or suite == "multitask" or tuple(algorithms) != ("ppo",)
    ):
        raise ValueError("optimized PPO runtime options support single/continual PPO training only")
    if compile_ppo and (phase != "train" or tuple(algorithms) != ("ppo",)):
        raise ValueError("compile_ppo supports PPO training only")
    if phase == "train":
        training_steps = steps if steps is not None else DEFAULT_TRAINING_STEPS[suite]
        if training_steps <= 0:
            raise ValueError("steps must be positive")
        budgets = task_steps if task_steps is not None else (training_steps,)
        if suite in {"single", "continual"} and "ppo" in algorithms:
            if any(value % num_envs for value in budgets):
                raise ValueError("PPO task budgets must be divisible by num_envs")
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
        if deterministic:
            jobs = [replace(job, arguments=(*job.arguments, "--deterministic")) for job in jobs]
        return jobs

    jobs = _build_evaluation_jobs(
        suite, games, algorithms, episodes, max_steps, ewc_lambda, methods, seed
    )
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
                name = f"single {game} {algorithm}"
                arguments = [
                    "scripts/train_single.py",
                    "--seed",
                    str(seed),
                    "--game",
                    game,
                    "--algorithm",
                    algorithm,
                    "--steps",
                    str(steps),
                ]
                if num_envs > 1:
                    arguments.extend(("--num-envs", str(num_envs)))
                if algorithm == "ppo" and env_backend != "sync":
                    arguments.extend(("--env-backend", env_backend))
                if algorithm == "ppo" and compile_ppo:
                    arguments.append("--compile-ppo")
                jobs.append(
                    Job(
                        name,
                        tuple(arguments),
                        f"{_game_slug(game)}_{algorithm}_seed{seed}.log",
                        True,
                    )
                )
    elif suite == "continual":
        for algorithm in algorithms:
            for method in methods:
                arguments = [
                    "scripts/train_continual.py",
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
                if algorithm == "ppo" and env_backend != "sync":
                    arguments.extend(("--env-backend", env_backend))
                if algorithm == "ppo" and compile_ppo:
                    arguments.append("--compile-ppo")
                jobs.append(
                    Job(
                        f"continual {algorithm} {variant}",
                        tuple(arguments),
                        f"continual_{algorithm}_{variant}_seed{seed}.log",
                        True,
                    )
                )
    else:
        for algorithm in algorithms:
            jobs.append(
                Job(
                    f"multitask {algorithm}",
                    (
                        "scripts/train_multitask.py",
                        "--seed",
                        str(seed),
                        "--games",
                        *games,
                        "--algorithm",
                        algorithm,
                        "--steps",
                        str(steps),
                        *(("--compile-ppo",) if compile_ppo else ()),
                    ),
                    f"multitask_{algorithm}_seed{seed}.log",
                    True,
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
                run_name = f"{game}_{algorithm}"
                jobs.append(
                    Job(
                        f"evaluate single {game} {algorithm}",
                        (
                            "scripts/evaluate.py",
                            "--seed",
                            str(seed),
                            "--mode",
                            "single",
                            "--model",
                            f"checkpoints/single/{run_name}/seed-{seed}.pt",
                            "--algorithm",
                            algorithm,
                            "--game",
                            game,
                            "--episodes",
                            str(episodes),
                            "--max-steps",
                            str(max_steps),
                            "--json-out",
                            f"outputs/single/{run_name}/seed-{seed}/eval/metrics.json",
                        ),
                        f"evaluate_{_game_slug(game)}_{algorithm}_seed{seed}.log",
                        False,
                    )
                )
    elif suite == "continual":
        for algorithm in algorithms:
            for method in methods:
                use_ewc = method == "ewc"
                variant = f"{algorithm}_gpm" if method == "gpm" else f"{algorithm}_ewc{use_ewc}"
                arguments = [
                    "scripts/evaluate.py",
                    "--seed",
                    str(seed),
                    "--mode",
                    "continual",
                    "--model",
                    f"checkpoints/continual/{variant}/seed-{seed}.pt",
                    "--algorithm",
                    algorithm,
                    "--games",
                    *games,
                    "--episodes",
                    str(episodes),
                    "--max-steps",
                    str(max_steps),
                    "--json-out",
                    f"outputs/continual/{variant}/seed-{seed}/eval/metrics.json",
                ]
                if use_ewc:
                    arguments.extend(("--ewc", "--ewc-lambda", str(ewc_lambda)))
                jobs.append(
                    Job(
                        f"evaluate continual {algorithm} {method}",
                        tuple(arguments),
                        f"evaluate_continual_{variant}_seed{seed}.log",
                        False,
                    )
                )
    else:
        for algorithm in algorithms:
            jobs.append(
                Job(
                    f"evaluate multitask {algorithm}",
                    (
                        "scripts/evaluate.py",
                        "--seed",
                        str(seed),
                        "--mode",
                        "multitask",
                        "--model",
                        f"checkpoints/multitask/{algorithm}/seed-{seed}.pt",
                        "--algorithm",
                        algorithm,
                        "--games",
                        *games,
                        "--episodes",
                        str(episodes),
                        "--max-steps",
                        str(max_steps),
                        "--json-out",
                        f"outputs/multitask/{algorithm}/seed-{seed}/eval/metrics.json",
                    ),
                    f"evaluate_multitask_{algorithm}_seed{seed}.log",
                    False,
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
    compile_ppo: bool = True,
    deterministic: bool = True,
) -> list[Job]:
    """Run all ten configurations; training includes a final evaluation of each job.

    Single-task uses the two teaching games. Joint training gets the same total
    transition budget as the original three-task protocol. Without a uniform steps
    override, Pong single-task uses 2M and continual PPO uses the verified budgets.
    """
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive")
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
                games=DEFAULT_GAMES[suite],
                algorithms=(algorithm,),
                seed=seed,
                episodes=episodes,
                max_steps=max_steps,
                method=method,
                ewc_lambda=ewc_lambda,
            )
            if phase == "train":
                vector = algorithm == "ppo" and suite != "multitask"
                budgets = (
                    (1_000_448, 500_000, 500_000)
                    if (steps is None and suite == "continual" and algorithm == "ppo")
                    else None
                )
                case_steps = steps or 500_000
                if suite == "multitask":
                    case_steps *= len(DEFAULT_GAMES[suite])
                training.extend(
                    build_jobs(
                        "train",
                        suite,
                        **options,
                        steps=None if budgets else case_steps,
                        task_steps=budgets,
                        num_envs=num_envs if vector else 1,
                        env_backend=env_backend if vector else "sync",
                        compile_ppo=compile_ppo and algorithm == "ppo",
                        deterministic=deterministic,
                    )
                )
            evaluation.extend(build_jobs("evaluate", suite, **options, deterministic=deterministic))
    if steps is None:
        updated = []
        for job in training:
            arguments = list(job.arguments)
            if arguments[0] == "scripts/train_single.py" and "Pong-v5" in arguments:
                arguments[arguments.index("--steps") + 1] = "2000000"
            if arguments[0] in {"scripts/train_single.py", "scripts/train_continual.py"}:
                arguments.extend(("--eval-interval", "250000"))
                if arguments[0] == "scripts/train_single.py":
                    arguments.extend(("--eval-episodes", str(episodes)))
            updated.append(replace(job, arguments=tuple(arguments)))
        training = updated
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
) -> None:
    """Execute jobs sequentially or concurrently and fail on any child error."""
    device_count = _cuda_device_count(device)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if not parallel:
        status_path = log_dir / "status.json"
        status = {
            "cwd": str(PROJECT_ROOT),
            "pid": os.getpid(),
            "jobs": [
                {
                    "name": job.name,
                    "command": list(job.command),
                    "log": str(log_dir / job.log_name),
                    "state": "pending",
                }
                for job in jobs
            ],
        }

        def write_status():
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(status_path)

        write_status()
        for index, job in enumerate(jobs):
            record = status["jobs"][index]
            record.update(state="running", started_at=datetime.now(timezone.utc).isoformat())
            write_status()
            environment = _job_environment(device_count, index, parallel=False)
            device_label = "cpu" if device_count == 0 else "cuda:0"
            _print_job(job, index + 1, len(jobs), device_label)
            if job.capture_output:
                log_path = log_dir / job.log_name
                with log_path.open("w", encoding="utf-8") as log_file:
                    result = subprocess.run(
                        job.command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
            else:
                result = subprocess.run(
                    job.command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                )
            record.update(
                state="completed" if result.returncode == 0 else "failed",
                exit_code=result.returncode,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            write_status()
            if result.returncode != 0:
                raise RuntimeError(f"Job failed with exit code {result.returncode}: {job.name}")
        return

    running = []
    try:
        for index, job in enumerate(jobs):
            environment = _job_environment(device_count, index, parallel=True)
            gpu_index = index % device_count if device_count else None
            device_label = "cpu" if gpu_index is None else f"cuda:{gpu_index}"
            _print_job(job, index + 1, len(jobs), device_label)
            log_path = log_dir / job.log_name
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                job.command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running.append((job, process, log_file))

        failures = []
        for job, process, _ in running:
            return_code = process.wait()
            if return_code != 0:
                failures.append(f"{job.name} (exit {return_code})")
        if failures:
            raise RuntimeError("Failed jobs: " + ", ".join(failures))
    except KeyboardInterrupt:
        for _, process, _ in running:
            process.terminate()
        for _, process, _ in running:
            process.wait()
        raise
    finally:
        for _, _, log_file in running:
            log_file.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("suite", choices=("single", "continual", "multitask", "teaching"))
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
        default=1,
        help="Environments used for batched PPO policy inference",
    )
    parser.add_argument(
        "--env-backend",
        choices=("sync", "async"),
        default="sync",
        help="PPO environment execution backend",
    )
    parser.add_argument(
        "--compile-ppo",
        action="store_true",
        help="Compile PPO rollout and learner policy evaluation",
    )
    parser.add_argument("--parallel", action="store_true", help="Run all jobs concurrently")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the experiment matrix and launch its Python child processes."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = args.seeds or (args.seed,)
        if args.suite == "teaching":
            if (
                args.games
                or args.method
                or args.task_steps
                or args.ewc_mode != "both"
                or args.parallel
            ):
                raise ValueError("teaching uses fixed games/methods and sequential execution")
            if tuple(args.algorithms) != ("dqn", "ppo"):
                raise ValueError("teaching includes both DQN and PPO")
            jobs = []
            for seed in seeds:
                jobs.extend(
                    build_teaching_jobs(
                        phase=args.phase,
                        seed=seed,
                        steps=args.steps,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                        ewc_lambda=args.ewc_lambda,
                        num_envs=args.num_envs,
                        env_backend=args.env_backend,
                        compile_ppo=args.compile_ppo,
                        deterministic=True if args.deterministic is None else args.deterministic,
                    )
                )
            if args.dry_run:
                for index, job in enumerate(jobs, start=1):
                    _print_job(job, index, len(jobs), args.device)
            else:
                run_jobs(jobs, device=args.device, parallel=False, log_dir=args.log_dir)
            return
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
                    num_envs=args.num_envs,
                    env_backend=args.env_backend,
                    compile_ppo=args.compile_ppo,
                    deterministic=args.deterministic,
                    method=args.method,
                    task_steps=args.task_steps,
                )
            )
        if args.dry_run:
            for index, job in enumerate(jobs, start=1):
                _print_job(job, index, len(jobs), args.device)
            return
        run_jobs(jobs, device=args.device, parallel=args.parallel, log_dir=args.log_dir)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
