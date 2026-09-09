"""Training entry points, transition budgets, evaluation, and reproducibility."""

import random
from inspect import signature

import numpy as np
import pytest
import torch

from algorithms import DEFAULT_DQN_LEARNING_STARTS
from scripts.train_continual import train_continual
from scripts.train_multitask import train_multitask
from scripts.train_single import train_single_game
from training import seed_everything


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


@pytest.mark.parametrize("compile_ppo", (False, True))
def test_joint_ppo_uses_shared_learner_and_preserves_budget(
    monkeypatch, tmp_path, compile_ppo, fresh_compiler_state
):
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
        output_dir=tmp_path / "case",
        games=["pong", "breakout"],
        algorithm="ppo",
        total_steps=134,
        batch_size=4,
        compile_ppo=compile_ppo,
    )
    assert updates == [("pong", 128, compile_ppo), ("breakout", 6, compile_ppo)]
    assert all(env.closed for env in environments.values())
    assert sum(env.transitions for env in environments.values()) == 134
    assert (tmp_path / "case/checkpoints/final.pt").exists()


def test_seed_everything_can_reset_deterministic_kernels():
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    seed_everything(0)
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
@pytest.mark.parametrize("num_envs", (1, 8))
def test_joint_records_actual_environment_steps(monkeypatch, tmp_path, algorithm, num_envs):
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
    total_steps = 131 if num_envs == 1 else 152
    # Check the video connection once; the other cases exercise only step accounting.
    save_video = algorithm == "ppo" and num_envs == 8
    module.train_multitask(
        output_dir=tmp_path / "case",
        games=["a", "b"],
        algorithm=algorithm,
        total_steps=total_steps,
        num_envs=num_envs,
        save_video=save_video,
    )
    summary = json.loads((tmp_path / "case/training_summary.json").read_text())
    assert summary["total_steps"] == total_steps
    assert summary["task_steps"] == {
        game: env.steps * num_envs for game, env in environments.items()
    }
    assert sum(summary["task_steps"].values()) == total_steps
    import imageio_ffmpeg

    if save_video:
        for game, env in environments.items():
            if env.steps:
                path = tmp_path / f"case/videos/{game}/training.mp4"
                frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
                assert frames == (env.steps + 1) // 2


@pytest.mark.parametrize("num_envs", (1, 8))
@pytest.mark.parametrize("eval_points", (0, 3, 10))
def test_single_periodic_evaluation_keeps_training_budget(
    monkeypatch, tmp_path, num_envs, eval_points
):
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
    # One vector case also checks recording across periodic evaluations.
    save_video = num_envs == 8 and eval_points == 3
    module.train_single_game(
        output_dir=tmp_path / "case",
        algorithm="dqn",
        num_steps=10 * num_envs,
        eval_interval=0 if eval_points else 4 * num_envs,
        eval_points=eval_points,
        eval_episodes=2,
        num_envs=num_envs,
        save_video=save_video,
    )
    data = json.loads((tmp_path / "case/learning.json").read_text())
    assert [p.name for p in (tmp_path / "case/checkpoints").glob("*.pt")] == ["final.pt"]
    assert all("checkpoint" not in row for row in data["evaluations"])
    expected = (
        [((10 * i + eval_points - 1) // eval_points) * num_envs for i in range(1, eval_points + 1)]
        if eval_points
        else [4 * num_envs, 8 * num_envs, 10 * num_envs]
    )
    assert [e["step"] for e in data["evaluations"]] == expected
    assert all(e["rewards"] == [3.0, 3.0] for e in data["evaluations"])
    assert environments[0].steps == 10
    assert all(env.closed for env in environments)
    import imageio_ffmpeg

    if save_video:
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(tmp_path / "case/videos/training.mp4"))
        assert frames == 5
