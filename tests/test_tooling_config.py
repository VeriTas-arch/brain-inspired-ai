"""Tests for repository-local tool cache configuration."""

import tomllib
from pathlib import Path


def test_pytest_and_ruff_caches_are_kept_under_dot_cache() -> None:
    with Path("pyproject.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["tool"]["pytest"]["ini_options"]["cache_dir"] == ".cache/pytest"
    assert config["tool"]["ruff"]["cache-dir"] == ".cache/ruff"
