"""Tests for the shared random-seed contract."""

import random

import numpy as np
import pytest
import torch

from utils import seed_everything


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
