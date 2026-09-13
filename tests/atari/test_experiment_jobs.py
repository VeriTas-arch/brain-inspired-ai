"""Experiment matrices, configuration, and command forwarding."""

import pytest

from biai.atari.scripts.evaluate import DEFAULT_MAX_EPISODE_STEPS as EVALUATE_MAX_EPISODE_STEPS
from biai.atari.scripts.run_experiments import (
    DEFAULT_GAMES,
    DEFAULT_MAX_EPISODE_STEPS,
    build_jobs,
    build_teaching_jobs,
    main,
)


def _default_jobs(phase: str, suite: str):
    return build_jobs(
        phase,
        suite,
        games=DEFAULT_GAMES[suite],
        algorithms=("dqn", "ppo"),
    )


def test_default_training_matrices_preserve_case_order_and_budgets() -> None:
    single = _default_jobs("train", "single")
    continual = _default_jobs("train", "continual")
    multitask = _default_jobs("train", "multitask")

    assert len(single) == 4
    assert [job.name for job in single] == [
        "single Pong-v5 dqn",
        "single Pong-v5 ppo",
        "single Breakout-v5 dqn",
        "single Breakout-v5 ppo",
    ]
    assert all(job.arguments[-1] == "500000" for job in single)

    assert len(continual) == 4
    assert sum("--use-ewc" in job.arguments for job in continual) == 2
    assert all("500000" in job.arguments for job in continual)

    assert len(multitask) == 2
    assert all(job.arguments[-1] == "50000" for job in multitask)


def test_default_evaluation_matrices_preserve_checkpoint_and_output_paths() -> None:
    single = _default_jobs("evaluate", "single")
    continual = _default_jobs("evaluate", "continual")
    multitask = _default_jobs("evaluate", "multitask")

    assert (len(single), len(continual), len(multitask)) == (4, 4, 2)
    assert "single-dqn-pong/checkpoints/final.pt" in single[0].arguments
    assert "single-dqn-pong/evaluation/evaluation.json" in single[0].arguments
    assert "continual-ppo-ewc/checkpoints/final.pt" in continual[-1].arguments
    assert "--ewc" in continual[-1].arguments
    assert "multitask-dqn/checkpoints/final.pt" in multitask[0].arguments
    max_steps_index = single[0].arguments.index("--max-steps")
    assert single[0].arguments[max_steps_index + 1] == str(DEFAULT_MAX_EPISODE_STEPS)
    assert DEFAULT_MAX_EPISODE_STEPS == EVALUATE_MAX_EPISODE_STEPS


def test_cpu_dry_run_prints_commands_without_launching(capsys) -> None:
    main(("train", "multitask", "--device", "cpu", "--dry-run"))

    output = capsys.readouterr().out
    assert "-m biai.atari.scripts.train_multitask" in output
    assert "multitask dqn (cpu)" in output
    assert "multitask ppo (cpu)" in output


def test_seed_is_forwarded_to_training_and_evaluation_jobs() -> None:
    training = build_jobs(
        "train",
        "continual",
        games=DEFAULT_GAMES["continual"],
        algorithms=("ppo",),
        seed=17,
    )
    evaluation = build_jobs(
        "evaluate",
        "continual",
        games=DEFAULT_GAMES["continual"],
        algorithms=("ppo",),
        seed=17,
    )

    for job in [*training, *evaluation]:
        seed_index = job.arguments.index("--seed")
        assert job.arguments[seed_index + 1] == "17"
        assert "seed17" in job.log_name


def test_deterministic_training_option_is_forwarded():
    jobs = build_jobs(
        "train", "continual", games=("Pong-v5",), algorithms=("ppo",), deterministic=True
    )
    assert all("--deterministic" in job.arguments for job in jobs)


def test_multi_seed_dry_run_expands_without_path_collisions(capsys) -> None:
    main(("train", "continual", "--algorithms", "ppo", "--seeds", "7", "11", "--dry-run"))

    output = capsys.readouterr().out
    assert output.count("continual ppo no_ewc") == 2
    assert output.count("continual ppo ewc") == 2
    assert "--seed 7" in output
    assert "--seed 11" in output


@pytest.mark.parametrize(
    "suite,algorithm,backend,compiled",
    (
        ("single", "dqn", "ale", False),
        ("multitask", "dqn", "ale", False),
        ("multitask", "ppo", "ale", False),
        ("continual", "ppo", "sync", False),
        ("continual", "ppo", "async", True),
        ("multitask", "ppo", "async", True),
    ),
)
def test_runtime_options_reach_supported_training_protocols(suite, algorithm, backend, compiled):
    jobs = build_jobs(
        "train",
        suite,
        games=DEFAULT_GAMES[suite],
        algorithms=(algorithm,),
        num_envs=8,
        env_backend=backend,
        env_threads=2,
        compile_ppo=compiled,
    )
    for job in jobs:
        for flag, expected in (
            ("--num-envs", "8"),
            ("--env-threads", "2"),
        ):
            assert job.arguments[job.arguments.index(flag) + 1] == expected
        actual_backend = (
            job.arguments[job.arguments.index("--env-backend") + 1]
            if "--env-backend" in job.arguments
            else "sync"
        )
        assert actual_backend == backend
        assert ("--compile-ppo" in job.arguments) == compiled


@pytest.mark.parametrize("suite", ("single", "continual"))
def test_training_forwards_complete_episode_step_limit(suite) -> None:
    jobs = build_jobs(
        "train",
        suite,
        games=DEFAULT_GAMES[suite],
        algorithms=("ppo",),
        max_steps=12_345,
    )

    for job in jobs:
        limit_index = job.arguments.index("--eval-max-steps")
        assert job.arguments[limit_index + 1] == "12345"


@pytest.mark.parametrize("compiled_algorithm", ("ppo", "dqn"))
def test_compilation_flags_reject_mixed_algorithm_matrix(compiled_algorithm):
    with pytest.raises(ValueError, match=f"{compiled_algorithm.upper()} only"):
        build_jobs(
            "train",
            "single",
            games=("Pong-v5",),
            algorithms=("dqn", "ppo"),
            **{f"compile_{compiled_algorithm}": True},
        )


@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
@pytest.mark.parametrize("method", ("finetune", "ewc", "gpm"))
def test_teaching_method_selects_one_matching_training_and_evaluation_job(method, algorithm):
    options = dict(games=DEFAULT_GAMES["continual"], algorithms=(algorithm,), method=method)
    training = build_jobs("train", "continual", task_steps=(1048576, 524288, 524288), **options)
    evaluation = build_jobs("evaluate", "continual", **options)
    assert len(training) == len(evaluation) == 1
    assert "biai.atari.scripts.train_continual" in training[0].arguments
    index = training[0].arguments.index("--task-steps")
    assert training[0].arguments[index + 1 : index + 4] == ("1048576", "524288", "524288")
    assert f"continual-{algorithm}-{method}/checkpoints/final.pt" in evaluation[0].arguments
    assert ("--ewc" in evaluation[0].arguments) == (method == "ewc")


def test_gpm_dry_run_uses_complete_teaching_entry_point(capsys):
    main(("train", "continual", "--algorithms", "ppo", "--method", "gpm", "--dry-run"))
    output = capsys.readouterr().out
    assert "biai.atari.scripts.train_continual" in output
    assert "--method gpm" in output


@pytest.mark.parametrize(
    "options",
    [
        {"method": "gpm", "ewc_mode": "on"},
        {"task_steps": (8, 8)},
        {"task_steps": (8, 0, 8)},
        {"task_steps": (8, 10, 8), "num_envs": 8},
        {"task_steps": (8, 8, 8), "steps": 8},
    ],
)
def test_runner_rejects_unsupported_method_and_task_budgets(options):
    with pytest.raises(ValueError):
        build_jobs(
            "train",
            "continual",
            **{"games": DEFAULT_GAMES["continual"], "algorithms": ("ppo",), **options},
        )


def test_teaching_matrix_preserves_formal_configuration_and_matching_evaluation():
    expected_budgets = {
        ("single", "Pong-v5"): ("--steps", ("2000000",)),
        ("single", "Breakout-v5"): ("--steps", ("500000",)),
        ("multitask", "dqn"): ("--steps", ("1500000",)),
        ("multitask", "ppo"): ("--steps", ("1500000",)),
        ("continual", "dqn"): ("--steps-per-game", ("500000",)),
        ("continual", "ppo"): ("--task-steps", ("1000448", "500000", "500000")),
    }
    jobs = build_teaching_jobs(phase="train", seed=0)
    training, evaluation = jobs[:12], jobs[12:]
    assert len(training) == len(evaluation) == 12
    assert len({job.log_name for job in jobs}) == 24
    assert all(job.capture_output and "--deterministic" in job.arguments for job in jobs)
    for job in jobs:
        algorithm = job.arguments[job.arguments.index("--algorithm") + 1]
        assert ("--compile-dqn" in job.arguments) == (algorithm == "dqn")
    for train, evaluate in zip(training, evaluation, strict=True):
        args = train.arguments
        algorithm = args[args.index("--algorithm") + 1]
        suite = train.name.split()[0]
        case = args[args.index("--game") + 1] if suite == "single" else algorithm
        flag, values = expected_budgets[suite, case]
        index = args.index(flag) + 1
        assert args[index : index + len(values)] == values
        assert args[args.index("--seed") + 1] == "0"
        assert args[args.index("--num-envs") + 1] == "8"
        assert args[args.index("--env-backend") + 1] == "async"
        assert ("--compile-ppo" in args) == (algorithm == "ppo")
        if suite != "multitask":
            assert args[args.index("--eval-points") + 1] == "10"
        assert evaluate.arguments[:2] == ("-m", "biai.atari.scripts.evaluate")
        assert evaluate.depends_on == (train.log_name,)
    assert sum("--method" in job.arguments and "gpm" in job.arguments for job in training) == 2


@pytest.mark.parametrize("optimized", (False, True))
def test_teaching_cli_preserves_formal_defaults_and_explicit_eager_override(monkeypatch, optimized):
    from biai.atari.scripts import run_experiments

    recorded = []
    monkeypatch.setattr(run_experiments, "_dispatch_jobs", lambda jobs, args: recorded.extend(jobs))
    options = (
        ()
        if optimized
        else (
            "--num-envs",
            "1",
            "--dqn-num-envs",
            "1",
            "--env-backend",
            "sync",
            "--no-compile-ppo",
            "--no-compile-dqn",
        )
    )
    main(("train", "teaching", *options))
    assert len(recorded) == 24
    for job in recorded:
        algorithm = job.arguments[job.arguments.index("--algorithm") + 1]
        assert ("--compile-dqn" in job.arguments) == (optimized and algorithm == "dqn")
    for job in recorded[:12]:
        algorithm = job.arguments[job.arguments.index("--algorithm") + 1]
        assert (f"--compile-{algorithm}" in job.arguments) == optimized
        assert ("--num-envs" in job.arguments) == optimized
        if optimized:
            assert job.arguments[job.arguments.index("--num-envs") + 1] == "8"
            assert job.arguments[job.arguments.index("--env-backend") + 1] == "async"


def test_teaching_uniform_budget_override_omits_formal_evaluation_points():
    jobs = build_teaching_jobs(phase="train", seed=0, steps=1024)[:12]
    assert all("--task-steps" not in job.arguments for job in jobs)
    assert all("--eval-points" not in job.arguments for job in jobs)


def test_teaching_evaluations_keep_deterministic_kernels_enabled():
    from biai.atari.scripts.run_experiments import build_teaching_jobs

    jobs = build_teaching_jobs(phase="evaluate", seed=0)
    assert len(jobs) == 12
    assert all("--deterministic" in job.arguments for job in jobs)


@pytest.mark.parametrize("smoke", (False, True))
def test_matched_teaching_matrix_has_all_baselines_and_exact_joint_quotas(smoke):
    jobs = build_teaching_jobs(phase="train", seed=0, matched_budget=True, smoke=smoke)
    assert len(jobs) == 28
    training, evaluation = jobs[:14], jobs[14:]
    for train, evaluate in zip(training, evaluation):
        assert evaluate.depends_on == (train.log_name,)
        assert train.case_id == evaluate.case_id
        arguments = train.arguments
        algorithm = arguments[arguments.index("--algorithm") + 1]
        quota = (2048 if algorithm == "ppo" else 10032) if smoke else 500000
        if "train_continual" in arguments[1]:
            assert arguments[arguments.index("--steps-per-game") + 1] == str(quota)
            assert "--task-steps" not in arguments
        else:
            total = int(arguments[arguments.index("--steps") + 1])
            if "train_multitask" in arguments[1]:
                assert total == 3 * quota
                assert arguments[arguments.index("--steps-per-game") + 1] == str(quota)
            else:
                assert total == quota
    assert (
        len(
            [
                job
                for job in training
                if "SpaceInvaders-v5" in job.arguments and "train_single" in job.arguments[1]
            ]
        )
        == 2
    )


def test_matched_cli_dry_run_and_scope(capsys):
    main(("train", "teaching", "--matched-budget", "--dry-run"))
    assert "[28/28]" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(("train", "single", "--matched-budget", "--dry-run"))
