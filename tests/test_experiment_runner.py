"""Tests for the unified Python experiment runner."""

from scripts.run_experiments import DEFAULT_GAMES, _job_environment, build_jobs, main


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
