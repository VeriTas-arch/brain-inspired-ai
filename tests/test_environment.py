"""Integration tests for the Atari preprocessing contract."""

import numpy as np
import pytest
import torch

from environments import AtariEnv


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
