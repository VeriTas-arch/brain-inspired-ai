"""Tests for PPO rollout-buffer boundaries and storage."""

import pytest
import torch

from utils import RolloutBuffer


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
