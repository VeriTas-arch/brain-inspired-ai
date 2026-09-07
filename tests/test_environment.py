"""Integration tests for the Atari preprocessing contract."""

import numpy as np
import pytest
import torch

from environments import AtariEnv, SyncVectorAtariEnv


def wrapper_names(env: AtariEnv) -> list[str]:
    names = []
    wrapped = env.env
    while hasattr(wrapped, "env"):
        names.append(type(wrapped).__name__)
        wrapped = wrapped.env
    return names


def test_train_and_eval_use_one_frame_skip_and_different_episode_semantics() -> None:
    train_env = AtariEnv("Pong-v5", seed=7, training=True)
    eval_env = AtariEnv("Pong-v5", seed=7, training=False)
    try:
        train_wrappers = wrapper_names(train_env)
        eval_wrappers = wrapper_names(eval_env)

        assert train_env.env.unwrapped._frameskip == 1
        assert eval_env.env.unwrapped._frameskip == 1
        assert train_wrappers.count("MaxAndSkipEnv") == 1
        assert eval_wrappers.count("MaxAndSkipEnv") == 1
        assert "EpisodicLifeEnv" in train_wrappers
        assert "ClipRewardEnv" in train_wrappers
        assert "EpisodicLifeEnv" not in eval_wrappers
        assert "ClipRewardEnv" not in eval_wrappers

        train_state = train_env.reset()
        eval_state = eval_env.reset()
        assert train_state.shape == (4, 84, 84)
        assert train_state.dtype == torch.uint8
        torch.testing.assert_close(train_state, eval_state)
    finally:
        train_env.close()
        eval_env.close()


def test_invalid_action_is_not_silently_clipped() -> None:
    env = AtariEnv("Pong-v5", seed=7)
    try:
        env.reset()
        with pytest.raises(ValueError, match="outside"):
            env.step(env.action_space)
    finally:
        env.close()


def test_step_preserves_terminated_and_truncated_flags() -> None:
    class _ActionSpace:
        @staticmethod
        def contains(action: int) -> bool:
            return action == 0

    class _TruncatedEnv:
        action_space = _ActionSpace()

        @staticmethod
        def step(action: int):
            return np.zeros((4, 84, 84), dtype=np.uint8), 1.0, False, True, {}

    env = AtariEnv.__new__(AtariEnv)
    env.env = _TruncatedEnv()

    _, reward, terminated, truncated = env.step(0)

    assert reward == 1.0
    assert not terminated
    assert truncated


def test_sync_vector_environment_resets_only_finished_instances() -> None:
    class _FakeEnv:
        action_space = 2

        def __init__(self, value: int) -> None:
            self.value = value
            self.reset_count = 0

        def reset(self) -> torch.Tensor:
            self.reset_count += 1
            return torch.full((4, 84, 84), self.value + 10, dtype=torch.uint8)

        def step(self, action: int):
            return (
                torch.full((4, 84, 84), self.value, dtype=torch.uint8),
                float(action),
                self.value == 0,
                False,
            )

    env = SyncVectorAtariEnv.__new__(SyncVectorAtariEnv)
    env.envs = [_FakeEnv(0), _FakeEnv(1)]
    env.num_envs = 2
    env.action_space = 2

    states, rewards, terminated, truncated = env.step(torch.tensor([1, 0]))
    reset_states = env.reset_done(states, terminated | truncated)

    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(terminated, torch.tensor([True, False]))
    torch.testing.assert_close(reset_states[0], torch.full_like(reset_states[0], 10))
    torch.testing.assert_close(reset_states[1], states[1])
    assert env.envs[0].reset_count == 1
    assert env.envs[1].reset_count == 0
