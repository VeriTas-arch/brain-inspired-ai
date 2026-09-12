"""Deterministic evaluation, continual-learning reports, and plots."""

import json

import pytest
import torch

from biai.atari.scripts.evaluate import _infer_eval_dir_from_model_path, _set_agent_eval
from biai.atari.scripts.train_continual import build_evaluation_report
from biai.atari.scripts.visualize_results import plot_continual_results, plot_evaluation_results
from biai.atari.training import run_evaluation_episodes


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


@pytest.mark.parametrize("use_ewc", (False, True))
def test_multihead_evaluation_records_the_loaded_inner_agents_protocol(
    monkeypatch, tmp_path, use_ewc
):
    from biai.atari.algorithms import EWCWrapper
    from biai.atari.scripts import evaluate

    class Agent(_FakeAgent):
        def load(self, path):
            self.environment_protocol = "ale_native_v1"

        def set_task(self, game):
            pass

    class Env(_FakeEnv):
        def __init__(self, game, *, render_mode, training, seed, backend):
            super().__init__()
            assert backend == "ale"
            assert training is False

        def close(self):
            pass

    agent = Agent()
    if use_ewc:
        agent = EWCWrapper(agent)
        monkeypatch.setattr(EWCWrapper, "load", lambda self, path: self.agent.load(path))
    monkeypatch.setattr(evaluate, "_build_multihead_agent", lambda *args: agent)
    monkeypatch.setattr(evaluate, "AtariEnv", Env)
    monkeypatch.setattr(evaluate, "_record_example_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "_plot_multi_game_results", lambda *args: tmp_path / "plot.png")

    result = evaluate.evaluate_continual(
        model_path="native.pt",
        games=["game"],
        algorithm="dqn",
        episodes=2,
        max_steps=10,
        use_ewc=use_ewc,
        ewc_lambda=0.4,
        output_dir=tmp_path,
    )

    assert result["environment_protocol"] == "ale_native_v1"
    assert result["games"]["game"]["rewards"] == [3.0, 3.0]


@pytest.mark.parametrize("model_path", ("checkpoints/final.pt", "final.pt"))
def test_checkpoint_evaluations_use_one_stable_directory_per_case(tmp_path, model_path):
    case = tmp_path / "single-ppo-pong"
    assert _infer_eval_dir_from_model_path(str(case / model_path)) == case / "evaluation"


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
    metrics_path = tmp_path / "single-ppo-pong" / "evaluation" / "evaluation.json"
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

    for name in (".pending-case", "smoke", "performance"):
        ignored = tmp_path / name / "evaluation"
        ignored.mkdir(parents=True)
        (ignored / "evaluation.json").write_text("incomplete output")
    output_path = plot_evaluation_results(tmp_path, tmp_path / "evaluation.png")

    assert output_path.is_file()


def test_plot_continual_results_reads_score_matrix_without_cross_game_average(tmp_path) -> None:
    report_path = tmp_path / "continual-ppo-gpm" / "training_summary.json"
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


@pytest.mark.parametrize("mode", ("single", "multitask", "continual"))
def test_evaluation_preserves_requested_deterministic_kernels(monkeypatch, tmp_path, mode):
    from biai.atari.scripts import evaluate
    from biai.atari.training import seed_everything

    class Agent(_FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()

        def load(self, path):
            pass

        def set_task(self, game):
            pass

        def select_action(self, state, deterministic=False):
            assert torch.are_deterministic_algorithms_enabled()
            return super().select_action(state, deterministic)

    class Env(_FakeEnv):
        action_space = 2

        def __init__(self, *args, **kwargs):
            super().__init__()

        def close(self):
            pass

    monkeypatch.setattr(evaluate, "DQNAgent", Agent)
    monkeypatch.setattr(evaluate, "_build_multihead_agent", lambda *args: Agent())
    monkeypatch.setattr(evaluate, "AtariEnv", Env)
    monkeypatch.setattr(evaluate, "_record_example_video", lambda *args, **kwargs: None)
    try:
        kwargs = dict(
            model_path="example.pt",
            algorithm="dqn",
            episodes=2,
            max_steps=10,
            output_dir=tmp_path,
            deterministic=True,
        )
        if mode == "single":
            result = evaluate.evaluate_single(game="game", **kwargs)
        elif mode == "multitask":
            result = evaluate.evaluate_multitask(games=["game"], **kwargs)
        else:
            result = evaluate.evaluate_continual(
                games=["game"], use_ewc=False, ewc_lambda=0.4, **kwargs
            )
        assert result["deterministic"] is True
        rewards = result["rewards"] if mode == "single" else result["games"]["game"]["rewards"]
        assert rewards == [3.0, 3.0]
    finally:
        seed_everything(0)


@pytest.mark.parametrize("budget", (500_000, 1_000_448, 2_000_000))
@pytest.mark.parametrize("step_size", (8, 1024))
def test_evaluation_points_follow_budget_and_completed_updates(budget, step_size):
    from biai.atari.training import evaluation_schedule

    steps = evaluation_schedule(budget, points=10, step_size=step_size)
    assert len(steps) == len(set(steps)) == 10
    assert steps[-1] == budget
    for i, step in enumerate(steps, 1):
        assert budget * i / 10 <= step < budget * i / 10 + step_size
        assert step == budget or step % step_size == 0


def test_evaluation_schedule_rejects_conflicting_or_impossible_counts():
    from biai.atari.training import evaluation_schedule

    with pytest.raises(ValueError, match="not both"):
        evaluation_schedule(100, points=10, interval=20)
    with pytest.raises(ValueError, match="boundaries"):
        evaluation_schedule(2048, points=10, step_size=1024)
    with pytest.raises(ValueError, match="nonnegative"):
        evaluation_schedule(100, points=-1)
