"""Tests for Python functionality exposed through the teaching notebook."""

import math

from scripts.tutorial_examples import (
    run_dqn_update_demo,
    run_ewc_penalty_demo,
    run_gae_demo,
    runtime_summary,
)


def test_runtime_summary_reports_installed_frameworks() -> None:
    summary = runtime_summary()

    assert summary["torch"]
    assert summary["torchrl"]
    assert isinstance(summary["cuda_available"], bool)
    assert isinstance(summary["cuda_usable"], bool)
    if summary["cuda_available"]:
        assert summary["cuda_usable"] or summary["cuda_error"] is not None


def test_dqn_demo_runs_the_maintained_update_path() -> None:
    result = run_dqn_update_demo(batch_size=2)

    assert result["batch_shapes"]["states"] == (2, 4, 84, 84)
    assert math.isfinite(result["loss"])


def test_gae_demo_runs_the_maintained_estimator() -> None:
    result = run_gae_demo()

    assert len(result["advantages"]) == 3
    assert result["returns"][-1] == 1.0


def test_ewc_demo_uses_a_representative_batch_and_detects_weight_movement() -> None:
    result = run_ewc_penalty_demo(batch_size=1)

    assert result["penalty_at_reference"] == 0.0
    assert result["penalty_after_move"] > 0.0
