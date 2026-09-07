"""Tests for the repository's dependency policy."""

import tomllib
from pathlib import Path


def test_direct_dependencies_use_lower_bounds_only() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = [
        *config["project"]["dependencies"],
        *config["project"]["optional-dependencies"]["dev"],
    ]

    assert all(">=" in requirement for requirement in requirements)
    assert all("<" not in requirement and "==" not in requirement for requirement in requirements)


def test_torchrl_is_the_only_added_rl_framework() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = [requirement.lower() for requirement in config["project"]["dependencies"]]

    assert any(requirement.startswith("torchrl>=") for requirement in dependencies)
    assert not any("stable-baselines3" in requirement for requirement in dependencies)
    assert not any("envpool" in requirement for requirement in dependencies)
