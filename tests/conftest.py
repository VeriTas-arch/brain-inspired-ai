"""Keep runtime settings and compiler caches independent between tests."""

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def isolated_runtime_settings():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if workspace_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace_config


@pytest.fixture
def fresh_compiler_state():
    # Task switches and repeated updates inside one test still share a cache.
    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()
