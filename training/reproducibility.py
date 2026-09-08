"""Reproducibility helpers shared by training and evaluation entry points."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing deterministic kernels."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
