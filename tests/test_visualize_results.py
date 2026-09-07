"""Tests for visualization of current result artifacts."""

import json

from scripts.visualize_results import plot_continual_results, plot_evaluation_results


def test_plot_evaluation_results_reads_current_metrics_schema(tmp_path) -> None:
    metrics_path = tmp_path / "single" / "run" / "eval" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "single",
                "game": "Pong-v5",
                "algorithm": "dqn",
                "episodes": 2,
                "rewards": [1.0, 2.0],
                "avg_reward": 1.5,
            }
        ),
        encoding="utf-8",
    )

    output_path = plot_evaluation_results(tmp_path, tmp_path / "evaluation.png")

    assert output_path.is_file()


def test_plot_continual_results_reads_score_matrix_without_cross_game_average(tmp_path) -> None:
    report_path = tmp_path / "continual" / "run" / "continual_evaluation.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "score_units": "raw_environment_reward",
                "score_matrix": [
                    {"stage": 1, "after_task": "pong", "scores": {"pong": 10.0}},
                    {"stage": 2, "after_task": "breakout", "scores": {"pong": 7.0}},
                ],
                "per_task": {"pong": {"forgetting": 3.0}},
            }
        ),
        encoding="utf-8",
    )

    output_path = plot_continual_results(tmp_path, tmp_path / "continual.png")

    assert output_path.is_file()
