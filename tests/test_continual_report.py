"""Tests for continual-learning evaluation artifacts."""

from scripts.train_continual import build_evaluation_report


def test_report_keeps_task_scores_separate_and_computes_forgetting() -> None:
    report = build_evaluation_report(
        {
            "pong": [(1, 10.0), (2, 7.0), (3, 9.0)],
            "breakout": [(2, 2.0), (3, 5.0)],
            "space_invaders": [(3, 4.0)],
        },
        ["pong", "breakout", "space_invaders"],
    )

    assert report["score_matrix"][0]["scores"] == {
        "pong": 10.0,
        "breakout": None,
        "space_invaders": None,
    }
    assert report["per_task"]["pong"] == {
        "score_after_learning": 10.0,
        "best_score": 10.0,
        "final_score": 9.0,
        "forgetting": 1.0,
    }
    assert report["per_task"]["breakout"]["score_after_learning"] == 2.0
    assert "average_score" not in report
