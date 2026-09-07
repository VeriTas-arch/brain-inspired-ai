"""Compatibility entry point for running the pytest suite."""

from pathlib import Path

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).parent / "tests")]))
