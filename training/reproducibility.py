"""Reproducibility helpers shared by training and evaluation entry points."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed RNGs; optionally require deterministic kernels for matched diagnostics."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
