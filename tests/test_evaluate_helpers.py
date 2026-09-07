"""Tests for shared evaluation helpers."""

from pathlib import Path

import pytest
import torch

from scripts.evaluate import _infer_eval_dir_from_model_path, _set_agent_eval
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
