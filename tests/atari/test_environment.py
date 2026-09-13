"""Integration tests for the Atari preprocessing contract."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from biai.atari.environments import AsyncVectorAtariEnv, AtariEnv, SyncVectorAtariEnv
from biai.atari.environments.atari_wrappers import GrayscaleObservation


@pytest.mark.integration
def test_async_workers_match_sync_with_threads_and_close_cleanly(monkeypatch):
    import os
    from contextlib import ExitStack
    from threading import Event, Thread

    with ExitStack() as resources:
        stopped = Event()
        thread = Thread(target=stopped.wait)
        thread.start()
        resources.callback(thread.join)
        resources.callback(stopped.set)
        sync = SyncVectorAtariEnv("Pong-v5", 2, seed=7)
        resources.callback(sync.close)
        monkeypatch.setattr(os, "fork", lambda: pytest.fail("Must not fork a threaded parent"))
        async_env = AsyncVectorAtariEnv("Pong-v5", 2, seed=7)
        resources.callback(async_env.close)
        workers = tuple(async_env.env.processes)
        torch.testing.assert_close(async_env.reset(), sync.reset(), rtol=0, atol=0)
        for action in range(8):
            actions = torch.full((2,), action % sync.action_space, dtype=torch.long)
            actual = async_env.step_and_reset(actions)
            expected = sync.step_and_reset(actions)
            for name in vars(expected):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    assert all(not worker.is_alive() for worker in workers)
    assert not thread.is_alive()


def test_fast_grayscale_is_pixel_exact_with_gymnasium() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(256, size=(84, 84, 3), dtype=np.uint8)
    # Include every gray level and saturated primary colors, where truncation matters.
    image[:4] = np.resize(np.arange(256, dtype=np.uint8), (4, 84, 1))
    image[4:7] = np.eye(3, dtype=np.uint8)[:, None, :] * 255
    env = gym.Env()
    env.observation_space = gym.spaces.Box(0, 255, shape=image.shape, dtype=np.uint8)
    reference = gym.wrappers.GrayscaleObservation(env)
    candidate = GrayscaleObservation(env)
    before = image.copy()
    np.testing.assert_array_equal(candidate.observation(image), reference.observation(image))
    np.testing.assert_array_equal(image, before)


@pytest.mark.integration
@pytest.mark.parametrize("game", ("Pong-v5", "Breakout-v5", "SpaceInvaders-v5"))
def test_native_backend_preserves_final_frames_and_repeats_seeds(monkeypatch, game) -> None:
    from biai.atari.environments import NativeVectorAtariEnv

    make_vec = gym.make_vec

    def short_episodes(env_id, **kwargs):
        return make_vec(env_id, **(kwargs | {"max_num_frames_per_episode": 16}))

    monkeypatch.setattr(gym, "make_vec", short_episodes)
    traces = []
    for _ in range(2):
        env = NativeVectorAtariEnv(game, 2, seed=7, num_threads=2)
        try:
            trace = [env.reset().clone()]
            truncated_count = 0
            for _ in range(8):
                transition = env.step_and_reset(torch.zeros(2, dtype=torch.long))
                assert transition.observations.shape == (2, 4, 84, 84)
                assert transition.observations.dtype == torch.uint8
                if transition.truncated.any():
                    truncated_count += 1
                    assert not torch.equal(
                        transition.observations[transition.truncated],
                        transition.transition_observations[transition.truncated],
                    )
                trace.extend(
                    value.clone()
                    for value in (
                        transition.observations,
                        transition.transition_observations,
                        transition.rewards,
                        transition.terminated,
                        transition.truncated,
                    )
                )
            assert truncated_count > 0
            traces.append(trace)
        finally:
            env.close()
    for first, second in zip(*traces, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.integration
def test_native_evaluation_keeps_raw_rewards_full_episodes_and_cached_autoreset(monkeypatch):
    options = []
    make_vec = gym.make_vec

    def short_episodes(env_id, **kwargs):
        options.append(kwargs)
        return make_vec(env_id, **(kwargs | {"max_num_frames_per_episode": 16}))

    monkeypatch.setattr(gym, "make_vec", short_episodes)
    env = AtariEnv("SpaceInvaders-v5", seed=7, training=False, backend="ale")
    try:
        env.reset()
        assert not options[-1]["reward_clipping"] and not options[-1]["episodic_life"]
        for _ in range(20):
            _, _, terminated, truncated = env.step(0)
            if terminated or truncated:
                expected = env._autoreset_observation.clone()
                torch.testing.assert_close(env.reset(), expected, rtol=0, atol=0)
                assert env._autoreset_observation is None
                break
        else:
            raise AssertionError("Short native episode did not truncate")
        assert env.env.render().shape == (84, 84)
    finally:
        env.close()


def wrapper_names(env: AtariEnv) -> list[str]:
    names = []
    wrapped = env.env
    while hasattr(wrapped, "env"):
        names.append(type(wrapped).__name__)
        wrapped = wrapped.env
    return names


@pytest.mark.integration
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


@pytest.mark.integration
@pytest.mark.parametrize("backend", ("sync", "ale"))
def test_invalid_action_is_not_silently_clipped(backend) -> None:
    env = AtariEnv("Pong-v5", seed=7, backend=backend)
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
    env.backend = "sync"

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


def test_async_vector_environment_recovers_final_observation_after_autoreset() -> None:
    reset_observations = np.stack(
        (
            np.full((4, 84, 84), 10, dtype=np.uint8),
            np.full((4, 84, 84), 2, dtype=np.uint8),
        )
    )
    final_observation = np.full((4, 84, 84), 9, dtype=np.uint8)

    class _FakeAsyncEnvironment:
        @staticmethod
        def step_async(actions):
            np.testing.assert_array_equal(actions, np.array([0, 1]))

        @staticmethod
        def step_wait():
            return (
                reset_observations,
                np.array([1.0, 2.0]),
                np.array([False, False]),
                np.array([True, False]),
                {
                    "final_obs": np.array([final_observation, None], dtype=object),
                    "_final_obs": np.array([True, False]),
                },
            )

    env = AsyncVectorAtariEnv.__new__(AsyncVectorAtariEnv)
    env.env = _FakeAsyncEnvironment()
    env.num_envs = 2
    env.action_space = 2

    transition = env.step_and_reset(torch.tensor([0, 1]))

    torch.testing.assert_close(transition.observations, torch.as_tensor(reset_observations))
    torch.testing.assert_close(
        transition.transition_observations[0], torch.as_tensor(final_observation)
    )
    torch.testing.assert_close(
        transition.transition_observations[1], torch.as_tensor(reset_observations[1])
    )


def test_async_vector_environment_rejects_missing_final_observation() -> None:
    class _BrokenAsyncEnvironment:
        @staticmethod
        def step_async(actions):
            pass

        @staticmethod
        def step_wait():
            return (
                np.zeros((2, 4, 84, 84), dtype=np.uint8),
                np.zeros(2),
                np.array([True, False]),
                np.zeros(2, dtype=np.bool_),
                {},
            )

    env = AsyncVectorAtariEnv.__new__(AsyncVectorAtariEnv)
    env.env = _BrokenAsyncEnvironment()
    env.num_envs = 2

    with pytest.raises(RuntimeError, match="final observation"):
        env.step_and_reset(torch.tensor([0, 0]))


@pytest.mark.parametrize("failure", ("create", "close"))
def test_sync_vector_releases_other_environments_after_failure(monkeypatch, failure):
    from unittest.mock import Mock

    from biai.atari.environments import atari_env

    first, second = Mock(action_space=2), Mock(action_space=2)
    error = RuntimeError("environment failure")
    factory = Mock(side_effect=[first, error if failure == "create" else second])
    monkeypatch.setattr(atari_env, "AtariEnv", factory)
    if failure == "create":
        with pytest.raises(RuntimeError) as caught:
            SyncVectorAtariEnv("Pong-v5", 2)
    else:
        environment = SyncVectorAtariEnv("Pong-v5", 2)
        first.close.side_effect = error
        with pytest.raises(RuntimeError) as caught:
            environment.close()
        second.close.assert_called_once()
    assert caught.value is error
    first.close.assert_called_once()
