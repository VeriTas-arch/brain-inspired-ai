"""Tests for replay-buffer storage behavior."""

import numpy as np
import torch

from utils import ReplayBuffer


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
