"""Reproducibility helpers shared by training and evaluation entry points."""

import hashlib
import os
import random

import numpy as np
import torch


def parameter_digest(model: torch.nn.Module) -> str:
    """Identify the exact named parameter values, independently of their device."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
