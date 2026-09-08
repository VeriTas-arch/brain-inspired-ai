"""Dependency policy, tool configuration, and framework imports."""

import tomllib
from pathlib import Path


def test_direct_dependencies_use_lower_bounds_only() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = config["project"]["dependencies"]

    assert all(">=" in requirement for requirement in requirements)
    assert all("<" not in requirement and "==" not in requirement for requirement in requirements)


def test_torchrl_is_the_only_added_rl_framework() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = [requirement.lower() for requirement in config["project"]["dependencies"]]

    assert any(requirement.startswith("torchrl>=") for requirement in dependencies)
    assert not any("stable-baselines3" in requirement for requirement in dependencies)
    assert not any("envpool" in requirement for requirement in dependencies)


def test_compute_stack_uses_the_current_development_baseline() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    dependencies = config["project"]["dependencies"]
    assert "torch>=2.14" in dependencies
    assert "torchrl>=0.13.3" in dependencies
    assert "tensordict>=0.13" in dependencies


def test_pytest_and_ruff_caches_are_kept_under_dot_cache() -> None:
    with Path("pyproject.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["tool"]["pytest"]["ini_options"]["cache_dir"] == ".cache/pytest"
    assert config["tool"]["ruff"]["cache-dir"] == ".cache/ruff"


def test_torchrl_core_imports() -> None:
    import tensordict
    import torch
    import torchrl
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    assert torch.__version__
    assert torchrl.__version__
    assert tensordict.__version__
    assert ReplayBuffer is not None
    assert LazyTensorStorage is not None
