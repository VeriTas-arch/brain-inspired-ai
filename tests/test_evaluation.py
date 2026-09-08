"""Deterministic evaluation, continual-learning reports, and plots."""

import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate import _infer_eval_dir_from_model_path, _set_agent_eval
from scripts.train_continual import build_evaluation_report
from scripts.visualize_results import plot_continual_results, plot_evaluation_results
from training import run_evaluation_episodes


class _FakeAgent:
    def __init__(self) -> None:
        self.network = torch.nn.Linear(1, 1)
        self.deterministic_flags = []

    def select_action(self, state, deterministic: bool = False) -> int:
        self.deterministic_flags.append(deterministic)
        return 0


class _FakeEnv:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        self.steps = 0
        return torch.zeros(1)

    def step(self, action: int):
        self.steps += 1
        return torch.zeros(1), float(self.steps), self.steps == 2, False


def test_run_episodes_is_deterministic_and_honors_episode_boundaries() -> None:
    agent = _FakeAgent()

    rewards = run_evaluation_episodes(agent, _FakeEnv(), episodes=2, max_steps=10)

    assert rewards == [3.0, 3.0]
    assert agent.deterministic_flags == [True, True, True, True]


def test_run_episodes_rejects_incomplete_episode_at_max_steps() -> None:
    with pytest.raises(RuntimeError, match="did not finish within 1 steps"):
        run_evaluation_episodes(_FakeAgent(), _FakeEnv(), episodes=1, max_steps=1)


def test_set_agent_eval_switches_owned_modules() -> None:
    agent = _FakeAgent()
    agent.network.train()

    _set_agent_eval(agent)

    assert not agent.network.training


def test_seed_checkpoint_path_maps_to_seed_evaluation_directory() -> None:
    output = _infer_eval_dir_from_model_path(
        "checkpoints/continual/ppo_ewcTrue/seed-17.pt",
        "continual",
    )

    assert output == Path("outputs/continual/ppo_ewcTrue/seed-17/eval")


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
