"""Replay and rollout buffer storage, sampling, and boundaries."""

import numpy as np
import pytest
import torch

from training import ReplayBuffer, RolloutBuffer


def test_replay_buffer_keeps_pixels_uint8_until_sampling() -> None:
    buffer = ReplayBuffer(capacity=2)
    state = torch.randint(256, (4, 84, 84), dtype=torch.uint8)
    buffer.add(state, 1, 1.0, state.clone(), False)

    assert buffer.buffer[0]["state"].dtype == np.uint8
    batch = buffer.sample(1)
    assert batch["states"].dtype == torch.uint8
    assert batch["next_states"].dtype == torch.uint8
    torch.testing.assert_close(batch["states"][0], state)


def test_oversized_sample_returns_every_transition_once() -> None:
    buffer = ReplayBuffer(capacity=2)
    state = torch.zeros((4, 84, 84), dtype=torch.uint8)
    buffer.add(state, 0, 0.0, state, False)
    buffer.add(state, 1, 0.0, state, False)

    batch = buffer.sample(10)

    assert sorted(batch["actions"].tolist()) == [0, 1]


def test_local_sampling_generator_does_not_advance_training_rng() -> None:
    buffer = ReplayBuffer(capacity=2)
    state = torch.zeros((4, 84, 84), dtype=torch.uint8)
    buffer.add(state, 0, 0.0, state, False)
    buffer.add(state, 1, 0.0, state, False)

    np.random.seed(17)
    expected = np.random.random()
    np.random.seed(17)
    buffer.sample(1, rng=np.random.default_rng(23))

    assert np.random.random() == expected


def test_rollout_buffer_keeps_pixels_uint8_and_reuses_storage() -> None:
    buffer = RolloutBuffer(capacity=2)
    state = torch.randint(256, (4, 84, 84), dtype=torch.uint8)
    buffer.add(state, 1, 1.0, False, -0.5, 0.25)

    first_storage = buffer.states
    batch = buffer.get_batch()
    assert batch["states"].dtype == torch.uint8
    torch.testing.assert_close(batch["states"][0], state)
    assert buffer.ready_for_update(final=True)
    assert not buffer.is_full()

    buffer.reset()
    buffer.add(state, 0, 0.0, True, -0.2, 0.1)
    assert buffer.states is first_storage


def test_rollout_buffer_rejects_overflow_and_empty_batches() -> None:
    buffer = RolloutBuffer(capacity=1)
    with pytest.raises(ValueError, match="empty"):
        buffer.get_batch()

    state = torch.zeros((4, 84, 84), dtype=torch.uint8)
    buffer.add(state, 0, 0.0, False, 0.0, 0.0)
    assert buffer.ready_for_update()
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(state, 0, 0.0, False, 0.0, 0.0)


def test_rollout_buffer_preserves_vector_environment_axes() -> None:
    buffer = RolloutBuffer(capacity=2, num_envs=3)
    states = torch.randint(256, (3, 4, 84, 84), dtype=torch.uint8)
    buffer.add(
        states,
        torch.tensor([0, 1, 2]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([False, True, False]),
        torch.tensor([-0.1, -0.2, -0.3]),
        torch.tensor([0.1, 0.2, 0.3]),
    )

    batch = buffer.get_batch()
    assert batch["states"].shape == (1, 3, 4, 84, 84)
    assert batch["actions"].shape == (1, 3)
    torch.testing.assert_close(batch["states"][0], states)
    assert buffer.transition_count == 3


def test_rollout_buffer_can_store_policy_output_before_environment_result() -> None:
    buffer = RolloutBuffer(capacity=1, num_envs=2)
    states = torch.arange(4).reshape(2, 2)

    buffer.start_step(states, torch.tensor([0, 1]), torch.tensor([-0.1, -0.2]), torch.ones(2))
    states.fill_(99)
    buffer.finish_step(torch.tensor([1.0, 2.0]), torch.tensor([False, True]))

    batch = buffer.get_batch()
    torch.testing.assert_close(batch["states"][0], torch.arange(4).reshape(2, 2))
    torch.testing.assert_close(batch["rewards"][0], torch.tensor([1.0, 2.0]))
