"""Replay and rollout buffer storage, sampling, and boundaries."""

import numpy as np
import pytest
import torch

from biai.atari.training import ReplayBuffer, RolloutBuffer


def test_batched_replay_matches_scalar_writes_even_when_batch_exceeds_capacity():
    scalar, batched = ReplayBuffer(7, seed=42), ReplayBuffer(7, seed=42)
    for start, count in ((0, 3), (3, 6), (9, 12)):
        states = torch.arange(start, start + count, dtype=torch.uint8).reshape(count, 1, 1, 1)
        actions = torch.arange(count)
        rewards = torch.arange(count, dtype=torch.float32) / 3
        dones = actions % 2 == 0
        for index in range(count):
            scalar.add(
                states[index],
                int(actions[index]),
                float(rewards[index]),
                states[index] + 1,
                bool(dones[index]),
            )
        batched.add_batch(states, actions, rewards, states + 1, dones)
        states.fill_(255)
        actual = batched.sample(20)
        for key, expected in scalar.sample(20).items():
            torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("local_rng", (False, True))
def test_replay_sampling_preserves_ring_data_dtypes_and_rng(local_rng) -> None:
    from collections import deque

    reference = deque(maxlen=5)
    buffer = ReplayBuffer(capacity=5, seed=41)
    np.random.seed(41)
    for index in range(13):
        state = torch.full((4, 3, 3), index, dtype=torch.uint8)
        next_state = state + 1
        buffer.add(state, index, index / 8, next_state, index % 2 == 0)
        reference.append((state.clone(), index, index / 8, next_state.clone(), index % 2 == 0))
        # Stored transitions must survive shared observation-buffer reuse.
        state.fill_(255)
        next_state.fill_(255)

    assert buffer._storage["states"].dtype == np.uint8
    assert buffer._storage["next_states"].dtype == np.uint8
    expected_rng = np.random.default_rng(9 if local_rng else 41)
    actual_rng = np.random.default_rng(9) if local_rng else None
    for size in (3, 9, 1):
        rng_state = np.random.get_state()
        indices = expected_rng.choice(len(reference), min(size, len(reference)), replace=False)
        expected_next_random = np.random.random()
        np.random.set_state(rng_state)
        batch = buffer.sample(size, rng=actual_rng)
        assert np.random.random() == expected_next_random
        rows = [reference[index] for index in indices]
        for key, column, dtype in (
            ("states", 0, torch.uint8),
            ("actions", 1, torch.long),
            ("rewards", 2, torch.float32),
            ("next_states", 3, torch.uint8),
            ("dones", 4, torch.float32),
        ):
            expected = torch.stack([torch.as_tensor(row[column], dtype=dtype) for row in rows])
            torch.testing.assert_close(batch[key], expected, rtol=0, atol=0)
        batch["states"].fill_(254)
        assert not (buffer._storage["states"][: len(buffer)] == 254).any()


def test_replay_seed_is_reproducible_and_external_sampling_leaves_it_unchanged() -> None:
    buffers = [ReplayBuffer(64, seed=(17, 2, 1)) for _ in range(2)]
    for buffer in buffers:
        for index in range(64):
            state = torch.full((1,), index, dtype=torch.uint8)
            buffer.add(state, index, 0.0, state, False)
    buffers[0].sample(8, rng=np.random.default_rng(99))
    for _ in range(4):
        first, second = (buffer.sample(32)["actions"] for buffer in buffers)
        torch.testing.assert_close(first, second)
        assert first.unique().numel() == 32


def test_rollout_buffer_keeps_pixels_uint8_and_reuses_storage() -> None:
    buffer = RolloutBuffer(capacity=2)
    state = torch.randint(256, (4, 84, 84), dtype=torch.uint8)
    buffer.start_step(state, 1, -0.5, 0.25)
    buffer.finish_step(1.0, False)

    first_storage = buffer.states
    batch = buffer.get_batch()
    assert batch["states"].dtype == torch.uint8
    torch.testing.assert_close(batch["states"][0], state)
    assert not buffer.is_full()

    buffer.reset()
    buffer.start_step(state, 0, -0.2, 0.1)
    buffer.finish_step(0.0, True)
    assert buffer.states is first_storage


def test_rollout_buffer_rejects_overflow_and_empty_batches() -> None:
    buffer = RolloutBuffer(capacity=1)
    with pytest.raises(ValueError, match="empty"):
        buffer.get_batch()

    state = torch.zeros((4, 84, 84), dtype=torch.uint8)
    buffer.start_step(state, 0, 0.0, 0.0)
    buffer.finish_step(0.0, False)
    with pytest.raises(RuntimeError, match="full"):
        buffer.start_step(state, 0, 0.0, 0.0)


def test_rollout_buffer_preserves_vector_environment_axes() -> None:
    buffer = RolloutBuffer(capacity=2, num_envs=3)
    states = torch.randint(256, (3, 4, 84, 84), dtype=torch.uint8)
    buffer.start_step(
        states,
        torch.tensor([0, 1, 2]),
        torch.tensor([-0.1, -0.2, -0.3]),
        torch.tensor([0.1, 0.2, 0.3]),
    )
    buffer.finish_step(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False]))

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


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_rollout_buffer_owns_policy_outputs_before_sampler_storage_is_reused(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    buffer = RolloutBuffer(capacity=2, num_envs=2, policy_device=device)
    log_probs = torch.tensor([-0.1, -0.2], device=device)
    values = torch.tensor([1.0, 2.0], device=device)
    states = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
    buffer.start_step(states, torch.tensor([0, 1]), log_probs, values)
    log_probs.fill_(-9.0)
    values.fill_(99.0)
    buffer.finish_step(torch.ones(2), torch.zeros(2))
    batch = buffer.get_batch()

    assert batch["states"].device.type == batch["rewards"].device.type == "cpu"
    assert batch["log_probs"].device.type == batch["values"].device.type == device
    torch.testing.assert_close(batch["log_probs"].cpu(), torch.tensor([[-0.1, -0.2]]))
    torch.testing.assert_close(batch["values"].cpu(), torch.tensor([[1.0, 2.0]]))
