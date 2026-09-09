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


@pytest.mark.parametrize(
    "suite,algorithm", (("single", "dqn"), ("multitask", "ppo"), ("multitask", "dqn"))
)
def test_vector_environment_options_reach_all_training_protocols(suite, algorithm):
    jobs = build_jobs(
        "train",
        suite,
        games=DEFAULT_GAMES[suite],
        algorithms=(algorithm,),
        num_envs=8,
        env_backend="ale",
        env_threads=2,
    )
    for job in jobs:
        assert job.arguments[job.arguments.index("--num-envs") + 1] == "8"
        assert job.arguments[job.arguments.index("--env-backend") + 1] == "ale"
        assert job.arguments[job.arguments.index("--env-threads") + 1] == "2"


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


def test_compilation_flags_reject_mixed_algorithm_matrix() -> None:
    with pytest.raises(ValueError, match="PPO only"):
        build_jobs(
            "train",
            "single",
            games=DEFAULT_GAMES["single"],
            algorithms=("dqn", "ppo"),
            env_backend="async",
            compile_ppo=True,
        )


@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
@pytest.mark.parametrize("method", ("finetune", "ewc", "gpm"))
def test_teaching_method_selects_one_matching_training_and_evaluation_job(method, algorithm):
    options = dict(games=DEFAULT_GAMES["continual"], algorithms=(algorithm,), method=method)
    training = build_jobs("train", "continual", task_steps=(1048576, 524288, 524288), **options)
    evaluation = build_jobs("evaluate", "continual", **options)
    assert len(training) == len(evaluation) == 1
    assert "scripts/train_continual.py" in training[0].arguments
    index = training[0].arguments.index("--task-steps")
    assert training[0].arguments[index + 1 : index + 4] == ("1048576", "524288", "524288")
    variant = f"{algorithm}_gpm" if method == "gpm" else f"{algorithm}_ewc{method == 'ewc'}"
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
def test_compile_ppo_requires_ppo(train) -> None:
    with pytest.raises(ValueError, match="compile_ppo requires PPO"):
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


def test_multitask_ppo_compile_option_reaches_training_job():
    jobs = build_jobs(
        phase="train", suite="multitask", games=("Pong-v5",), algorithms=("ppo",), compile_ppo=True
    )
    assert len(jobs) == 1
    assert "--compile-ppo" in jobs[0].arguments


@pytest.mark.parametrize("compile_ppo", (False, True))
def test_joint_ppo_uses_shared_learner_and_preserves_budget(monkeypatch, tmp_path, compile_ppo):
    from algorithms import MultiHeadPPOAgent
    from scripts import train_multitask as joint

    if compile_ppo and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if compile_ppo else "cpu"
    environments, updates = {}, []

    class TinyAgent(MultiHeadPPOAgent):
        def __init__(self, state_dim, **kwargs):
            super().__init__(state_dim, device=device, **kwargs)
            self.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 512)).to(
                device
            )
            self.network = self.backbone

    class TinyEnvironment:
        action_space = 2

        def __init__(self, game, **kwargs):
            environments[game] = self
            self.transitions = 0
            self.closed = False

        def reset(self):
            return torch.ones(4, 1, 1, dtype=torch.uint8)

        def step(self, action):
            assert 0 <= action < self.action_space
            self.transitions += 1
            return self.reset(), 1.0, False, self.transitions % 3 == 0

        def close(self):
            self.closed = True

    update = joint.PPOLearner.update

    def record_update(learner, rollout):
        updates.append((learner.agent.current_task, rollout.transition_count, learner.compiled))
        return update(learner, rollout)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(joint, "AtariEnv", TinyEnvironment)
    monkeypatch.setattr("environments.atari_env.AtariEnv", TinyEnvironment)
    monkeypatch.setattr(joint, "MultiHeadPPOAgent", TinyAgent)
    monkeypatch.setattr(joint.PPOLearner, "update", record_update)
    monkeypatch.setattr(
        joint.np.random,
        "choice",
        lambda games: np.str_("pong" if environments["pong"].transitions < 128 else "breakout"),
    )
    joint.train_multitask(
        games=["pong", "breakout"],
        algorithm="ppo",
        total_steps=134,
        batch_size=4,
        compile_ppo=compile_ppo,
    )
    assert updates == [("pong", 128, compile_ppo), ("breakout", 6, compile_ppo)]
    assert all(env.closed for env in environments.values())
    assert sum(env.transitions for env in environments.values()) == 134
    assert (tmp_path / "checkpoints/multitask/ppo/seed-0.pt").exists()


def test_teaching_matrix_covers_ten_configurations_and_matching_evaluation():
    from scripts.run_experiments import build_teaching_jobs

    jobs = build_teaching_jobs(phase="train", seed=0)
    training, evaluation = jobs[:12], jobs[12:]
    assert len(training) == len(evaluation) == 12
    assert len({job.log_name for job in jobs}) == 24
    for job in training:
        assert job.arguments[job.arguments.index("--seed") + 1] == "0"
        algorithm = job.arguments[job.arguments.index("--algorithm") + 1]
        assert ("--compile-ppo" in job.arguments) == (algorithm == "ppo")
        if "multitask" in job.name:
            assert job.arguments[job.arguments.index("--steps") + 1] == "1500000"
    assert all(job.capture_output for job in jobs)
    assert sum("--method" in j.arguments and "gpm" in j.arguments for j in training) == 2
    assert all("scripts/evaluate.py" in job.arguments for job in evaluation)


@pytest.mark.parametrize("optimized", (False, True))
def test_teaching_cli_preserves_formal_defaults_and_explicit_eager_override(monkeypatch, optimized):
    from scripts import run_experiments

    recorded = []
    monkeypatch.setattr(run_experiments, "run_jobs", lambda jobs, **kwargs: recorded.extend(jobs))
    options = (
        ()
        if optimized
        else ("--num-envs", "1", "--env-backend", "sync", "--no-compile-ppo", "--no-compile-dqn")
    )
    main(("train", "teaching", *options))
    assert len(recorded) == 24
    for job in recorded[:12]:
        algorithm = job.arguments[job.arguments.index("--algorithm") + 1]
        assert (f"--compile-{algorithm}" in job.arguments) == optimized
        if algorithm == "ppo" and "multitask" not in job.name:
            assert ("--num-envs" in job.arguments) == optimized
            if optimized:
                assert job.arguments[job.arguments.index("--num-envs") + 1] == "8"
                assert job.arguments[job.arguments.index("--env-backend") + 1] == "async"


def test_sequential_runner_records_completion_failure_and_pending_jobs(tmp_path):
    import json

    from scripts.run_experiments import Job, run_jobs

    jobs = [
        Job("ok", ("-c", "print('done')"), "ok.log", True),
        Job("fail", ("-c", "raise SystemExit(3)"), "fail.log", True),
        Job("pending", ("-c", "print('not run')"), "pending.log", True),
    ]
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_jobs(jobs, device="cpu", parallel=False, log_dir=tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert [job["state"] for job in status["jobs"]] == ["completed", "failed", "pending"]
    assert status["jobs"][0]["exit_code"] == 0
    assert status["jobs"][1]["exit_code"] == 3
    assert (tmp_path / "ok.log").read_text().strip() == "done"


def test_bounded_runner_enforces_dependencies_and_cpu_quotas(tmp_path):
    import json

    from scripts.run_experiments import Job, run_jobs

    script = "import os,time,json; print(json.dumps({'cpus':len(os.sched_getaffinity(0))})); time.sleep(0.2)"
    jobs = [
        Job("eval a", ("-c", script), "eval-a.log", True, ("train-a.log",)),
        Job("train a", ("-c", script), "train-a.log", True),
        Job("train b", ("-c", script), "train-b.log", True),
        Job("eval b", ("-c", script), "eval-b.log", True, ("train-b.log",)),
    ]
    run_jobs(jobs, device="cpu", parallel=True, max_workers=2, cpus_per_job=1, log_dir=tmp_path)
    records = json.loads((tmp_path / "status.json").read_text())["jobs"]
    assert all(record["state"] == "completed" for record in records)
    assert records[0]["started_at"] >= records[1]["finished_at"]
    assert records[3]["started_at"] >= records[2]["finished_at"]
    events = sorted(
        [(r["started_at"], 1) for r in records] + [(r["finished_at"], -1) for r in records]
    )
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak == 2 and active == 0
    assert all(json.loads((tmp_path / job.log_name).read_text())["cpus"] == 1 for job in jobs)


def test_new_runs_freeze_source_and_keep_artifacts_separate(monkeypatch, tmp_path):
    import hashlib
    import json

    from scripts import run_experiments as runner

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    script = project / "scripts/train_example.py"
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    job = runner.Job("example", ("scripts/train_example.py",), "example.log", True)
    for value in ("first", "second"):
        script.write_text(f"from pathlib import Path\nPath('result.txt').write_text('{value}')\n")
        directory = tmp_path / value
        runner.run_jobs(
            [job], device="cpu", parallel=False, log_dir=directory / "logs", run_dir=directory
        )
        manifest = json.loads((directory / "source_manifest.json").read_text())
        assert (
            manifest["scripts/train_example.py"] == hashlib.sha256(script.read_bytes()).hexdigest()
        )
    assert (tmp_path / "first/result.txt").read_text() == "first"
    assert (tmp_path / "second/result.txt").read_text() == "second"
    assert "first" in (tmp_path / "first/source/scripts/train_example.py").read_text()
    with pytest.raises(ValueError, match="new run directory"):
        runner.run_jobs(
            [job],
            device="cpu",
            parallel=False,
            log_dir=tmp_path / "logs",
            run_dir=tmp_path / "first",
        )


def test_deterministic_training_option_is_forwarded_and_can_be_reset() -> None:
    jobs = build_jobs(
        "train", "continual", games=("Pong-v5",), algorithms=("ppo",), deterministic=True
    )
    assert all("--deterministic" in job.arguments for job in jobs)
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        seed_everything(0)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        seed_everything(0)


@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
def test_joint_records_actual_environment_steps(monkeypatch, tmp_path, algorithm):
    import importlib
    import json

    module = importlib.import_module("scripts.train_multitask")
    environments = {}

    class CountingEnvironment:
        action_space = 2

        def __init__(self, game, **kwargs):
            self.steps = 0
            environments[game] = self

        def reset(self):
            return torch.zeros(4, 84, 84, dtype=torch.uint8)

        def step(self, action):
            self.steps += 1
            return self.reset(), 0.0, False, False

        def close(self):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "AtariEnv", CountingEnvironment)
    monkeypatch.setattr("environments.atari_env.AtariEnv", CountingEnvironment)
    module.train_multitask(games=["a", "b"], algorithm=algorithm, total_steps=131)
    summary = json.loads(
        (tmp_path / f"outputs/multitask/{algorithm}/seed-0/training_summary.json").read_text()
    )
    assert summary["total_steps"] == 131
    assert summary["task_steps"] == {game: env.steps for game, env in environments.items()}
    assert sum(summary["task_steps"].values()) == 131


def test_single_periodic_evaluation_keeps_training_budget(monkeypatch, tmp_path):
    import importlib
    import json

    module = importlib.import_module("scripts.train_single")
    environments = []

    class CountingEnvironment:
        action_space = 2

        def __init__(self, game, training=True, **kwargs):
            self.training, self.steps, self.closed = training, 0, False
            environments.append(self)

        def reset(self):
            return torch.zeros(4, 84, 84, dtype=torch.uint8)

        def step(self, action):
            self.steps += 1
            return self.reset(), 3.0, not self.training, False

        def close(self):
            self.closed = True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "AtariEnv", CountingEnvironment)
    monkeypatch.setattr("environments.atari_env.AtariEnv", CountingEnvironment)
    module.train_single_game(algorithm="dqn", num_steps=10, eval_interval=4, eval_episodes=2)
    data = json.loads(
        (tmp_path / "outputs/single/Pong-v5_dqn/seed-0/learning_evaluation.json").read_text()
    )
    assert [e["step"] for e in data["evaluations"]] == [4, 8, 10]
    assert all(e["rewards"] == [3.0, 3.0] for e in data["evaluations"])
    assert all((tmp_path / e["checkpoint"]).is_file() for e in data["evaluations"])
    assert environments[0].steps == 10
    assert all(env.closed for env in environments)


def test_teaching_defaults_use_qualified_budgets_and_keep_uniform_smoke_override():
    from scripts.run_experiments import build_teaching_jobs

    jobs = build_teaching_jobs(phase="train", seed=0)[:12]
    for job in jobs:
        args = job.arguments
        assert "--deterministic" in args
        if args[0] == "scripts/train_single.py":
            expected = "2000000" if "Pong-v5" in args else "500000"
            assert args[args.index("--steps") + 1] == expected
            assert "--eval-interval" in args
        elif args[0] == "scripts/train_continual.py":
            if args[args.index("--algorithm") + 1] == "ppo":
                index = args.index("--task-steps")
                assert args[index + 1 : index + 4] == ("1000448", "500000", "500000")
            else:
                assert args[args.index("--steps-per-game") + 1] == "500000"
    smoke = build_teaching_jobs(phase="train", seed=0, steps=1024)[:12]
    assert all("--task-steps" not in j.arguments for j in smoke)
    assert all("--eval-interval" not in j.arguments for j in smoke)


def test_teaching_evaluations_keep_deterministic_kernels_enabled():
    from scripts.run_experiments import build_teaching_jobs

    jobs = build_teaching_jobs(phase="evaluate", seed=0)
    assert len(jobs) == 12
    assert all("--deterministic" in job.arguments for job in jobs)
