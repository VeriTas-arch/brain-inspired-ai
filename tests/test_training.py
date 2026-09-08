"""Training commands, defaults, validation, and reproducibility."""

import random
from inspect import signature

import numpy as np
import pytest
import torch

from algorithms import DEFAULT_DQN_LEARNING_STARTS
from scripts.evaluate import DEFAULT_MAX_EPISODE_STEPS as EVALUATE_MAX_EPISODE_STEPS
from scripts.run_experiments import (
    DEFAULT_GAMES,
    DEFAULT_MAX_EPISODE_STEPS,
    _job_environment,
    build_jobs,
    main,
)
from scripts.train_continual import train_continual
from scripts.train_multitask import train_multitask
from scripts.train_single import train_single_game
from training import seed_everything


def _default_jobs(phase: str, suite: str):
    return build_jobs(
        phase,
        suite,
        games=DEFAULT_GAMES[suite],
        algorithms=("dqn", "ppo"),
    )


def test_default_training_matrices_match_the_removed_shell_scripts() -> None:
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
    assert "checkpoints/single/Pong-v5_dqn/seed-0.pt" in single[0].arguments
    assert "outputs/single/Pong-v5_dqn/seed-0/eval/metrics.json" in single[0].arguments
    assert "checkpoints/continual/ppo_ewcTrue/seed-0.pt" in continual[-1].arguments
    assert "--ewc" in continual[-1].arguments
    assert "checkpoints/multitask/dqn/seed-0.pt" in multitask[0].arguments
    max_steps_index = single[0].arguments.index("--max-steps")
    assert single[0].arguments[max_steps_index + 1] == str(DEFAULT_MAX_EPISODE_STEPS)
    assert DEFAULT_MAX_EPISODE_STEPS == EVALUATE_MAX_EPISODE_STEPS


def test_cpu_dry_run_prints_commands_without_launching(capsys) -> None:
    main(("train", "multitask", "--device", "cpu", "--dry-run"))

    output = capsys.readouterr().out
    assert "train_multitask.py" in output
    assert "multitask dqn (cpu)" in output
    assert "multitask ppo (cpu)" in output


def test_child_device_assignment_supports_cpu_and_gpu_round_robin() -> None:
    assert _job_environment(0, job_index=3, parallel=True)["CUDA_VISIBLE_DEVICES"] == ""
    assert _job_environment(2, job_index=3, parallel=True)["CUDA_VISIBLE_DEVICES"] == "1"
    assert _job_environment(2, job_index=3, parallel=False)["CUDA_VISIBLE_DEVICES"] == "0"


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


def test_multi_seed_dry_run_expands_without_path_collisions(capsys) -> None:
    main(("train", "continual", "--algorithms", "ppo", "--seeds", "7", "11", "--dry-run"))

    output = capsys.readouterr().out
    assert output.count("continual ppo no_ewc") == 2
    assert output.count("continual ppo ewc") == 2
    assert "--seed 7" in output
    assert "--seed 11" in output


def test_vector_environment_count_is_forwarded_only_to_supported_ppo_training() -> None:
    jobs = build_jobs(
        "train",
        "continual",
        games=DEFAULT_GAMES["continual"],
        algorithms=("ppo",),
        num_envs=8,
    )

    for job in jobs:
        env_index = job.arguments.index("--num-envs")
        assert job.arguments[env_index + 1] == "8"


def test_vector_environments_reject_unsupported_training_matrices() -> None:
    for suite, algorithms in (("single", ("dqn",)), ("multitask", ("ppo",))):
        try:
            build_jobs(
                "train",
                suite,
                games=DEFAULT_GAMES[suite],
                algorithms=algorithms,
                num_envs=8,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported vector environment matrix must be rejected")


def test_optimized_ppo_runtime_options_are_forwarded() -> None:
    jobs = build_jobs(
        "train",
        "continual",
        games=DEFAULT_GAMES["continual"],
        algorithms=("ppo",),
        num_envs=8,
        env_backend="async",
        compile_ppo=True,
    )

    for job in jobs:
        assert "--env-backend" in job.arguments
        assert job.arguments[job.arguments.index("--env-backend") + 1] == "async"
        assert "--compile-ppo" in job.arguments


def test_continual_training_forwards_complete_episode_step_limit() -> None:
    jobs = build_jobs(
        "train",
        "continual",
        games=DEFAULT_GAMES["continual"],
        algorithms=("ppo",),
        max_steps=12_345,
    )

    for job in jobs:
        limit_index = job.arguments.index("--eval-max-steps")
        assert job.arguments[limit_index + 1] == "12345"


def test_optimized_ppo_runtime_rejects_mixed_algorithm_matrix() -> None:
    try:
        build_jobs(
            "train",
            "single",
            games=DEFAULT_GAMES["single"],
            algorithms=("dqn", "ppo"),
            env_backend="async",
        )
    except ValueError as error:
        assert "optimized PPO runtime" in str(error)
    else:
        raise AssertionError("optimized PPO options must not leak into DQN jobs")


@pytest.mark.parametrize("method", ("finetune", "ewc", "gpm"))
def test_teaching_method_selects_one_matching_training_and_evaluation_job(method):
    options = dict(games=DEFAULT_GAMES["continual"], algorithms=("ppo",), method=method)
    training = build_jobs("train", "continual", task_steps=(1048576, 524288, 524288), **options)
    evaluation = build_jobs("evaluate", "continual", **options)
    assert len(training) == len(evaluation) == 1
    assert "scripts/train_continual.py" in training[0].arguments
    index = training[0].arguments.index("--task-steps")
    assert training[0].arguments[index + 1 : index + 4] == ("1048576", "524288", "524288")
    variant = "ppo_gpm" if method == "gpm" else f"ppo_ewc{method == 'ewc'}"
    assert f"checkpoints/continual/{variant}/seed-0.pt" in evaluation[0].arguments
    assert ("--ewc" in evaluation[0].arguments) == (method == "ewc")


def test_gpm_dry_run_uses_complete_teaching_entry_point(capsys):
    main(("train", "continual", "--algorithms", "ppo", "--method", "gpm", "--dry-run"))
    output = capsys.readouterr().out
    assert "scripts/train_continual.py" in output
    assert "--method gpm" in output


@pytest.mark.parametrize(
    "options",
    [
        {"algorithms": ("dqn",), "method": "gpm"},
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


def test_continual_dqn_starts_learning_before_default_task_ends() -> None:
    default_task_steps = signature(train_continual).parameters["steps_per_game"].default

    assert DEFAULT_DQN_LEARNING_STARTS < default_task_steps


def test_continual_training_has_a_reproducible_default_seed() -> None:
    assert signature(train_continual).parameters["seed"].default == 0


@pytest.mark.parametrize(
    "train",
    (train_single_game, train_continual, train_multitask),
)
def test_training_entry_points_reject_invalid_batch_size(train) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        train(batch_size=0)


@pytest.mark.parametrize("train", (train_single_game, train_continual))
def test_ppo_vector_training_rejects_invalid_environment_counts(train) -> None:
    with pytest.raises(ValueError, match="num_envs must be positive"):
        train(algorithm="ppo", num_envs=0)


def test_ppo_vector_training_requires_divisible_step_budgets() -> None:
    with pytest.raises(ValueError, match="divisible"):
        train_single_game(algorithm="ppo", num_steps=10, num_envs=8)
    with pytest.raises(ValueError, match="divisible"):
        train_continual(algorithm="ppo", steps_per_game=10, num_envs=8)


@pytest.mark.parametrize("train", (train_single_game, train_continual))
def test_optimized_runtime_options_are_ppo_only(train) -> None:
    with pytest.raises(ValueError, match="only for PPO"):
        train(algorithm="dqn", env_backend="async")
    with pytest.raises(ValueError, match="only for PPO"):
        train(algorithm="dqn", compile_ppo=True)


def test_continual_training_rejects_invalid_evaluation_step_limit() -> None:
    with pytest.raises(ValueError, match="eval_max_steps must be positive"):
        train_continual(eval_max_steps=0)


def test_seed_everything_reproduces_python_numpy_and_torch() -> None:
    seed_everything(7)
    first = (random.random(), np.random.random(), torch.rand(1))
    seed_everything(7)
    second = (random.random(), np.random.random(), torch.rand(1))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_seed_everything_rejects_numpy_incompatible_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(2**32)
