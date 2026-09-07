"""Tests for the PPO runtime shared by training protocols."""

import torch

from environments import VectorStep
from training import PPOCollector, PPOLearner, flatten_rollout_data


class _FakeAgent:
    device = torch.device("cpu")
    gamma = 0.5
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 0.5

    @staticmethod
    def _values(states: torch.Tensor) -> torch.Tensor:
        return states[:, 0, 0, 0].to(torch.float32).unsqueeze(1)

    def sample_action_and_value(self, states: torch.Tensor):
        batch_size = len(states)
        return (
            torch.zeros(batch_size, dtype=torch.long),
            torch.zeros(batch_size),
            self._values(states),
        )

    def get_action_and_value(self, states: torch.Tensor, actions: torch.Tensor | None = None):
        if actions is None:
            actions = torch.zeros(len(states), dtype=torch.long)
        zeros = torch.zeros(len(states))
        return actions, zeros, zeros, self._values(states)

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        return self._values(states)

    @staticmethod
    def update(rollout_data, next_value, update_epochs, minibatch_size, **kwargs):
        assert kwargs["policy_evaluator"] is not None
        return {
            "transitions": float(len(rollout_data["actions"].flatten())),
            "next_value": float(next_value.sum()),
            "epochs": float(update_epochs),
            "minibatch_size": float(minibatch_size),
        }


class _FakeVectorEnvironment:
    num_envs = 2

    def __init__(self) -> None:
        self.step_index = 0

    @staticmethod
    def _observations(first: int, second: int) -> torch.Tensor:
        return torch.stack(
            (
                torch.full((4, 84, 84), first, dtype=torch.uint8),
                torch.full((4, 84, 84), second, dtype=torch.uint8),
            )
        )

    def reset(self) -> torch.Tensor:
        return self._observations(1, 2)

    def step_and_reset(self, actions: torch.Tensor) -> VectorStep:
        torch.testing.assert_close(actions, torch.zeros(2, dtype=torch.long))
        if self.step_index == 0:
            result = VectorStep(
                observations=self._observations(100, 3),
                transition_observations=self._observations(9, 3),
                rewards=torch.tensor([1.0, 2.0]),
                terminated=torch.tensor([False, False]),
                truncated=torch.tensor([True, False]),
            )
        else:
            result = VectorStep(
                observations=self._observations(4, 200),
                transition_observations=self._observations(4, 8),
                rewards=torch.tensor([3.0, 4.0]),
                terminated=torch.tensor([False, True]),
                truncated=torch.tensor([False, False]),
            )
        self.step_index += 1
        return result


def test_collector_preserves_vector_trajectories_and_time_limit_bootstrap() -> None:
    learner = PPOLearner(_FakeAgent(), update_epochs=3, minibatch_size=32)
    collector = PPOCollector(_FakeVectorEnvironment(), learner, rollout_length=8)

    rollout = collector.collect(transition_budget=4)

    assert rollout.transition_count == 4
    assert rollout.episode_returns == (1.0, 6.0)
    assert collector.episode_count == 2
    torch.testing.assert_close(
        rollout.data["rewards"],
        torch.tensor([[5.5, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        rollout.data["dones"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert rollout.data["states"][0, 0, 0, 0, 0].item() == 1
    assert rollout.data["states"][1, 0, 0, 0, 0].item() == 100
    torch.testing.assert_close(rollout.next_value, torch.tensor([4.0, 200.0]))

    metrics = learner.update(rollout)
    assert metrics == {
        "transitions": 4.0,
        "next_value": 204.0,
        "epochs": 3.0,
        "minibatch_size": 32.0,
    }


def test_collector_requires_exact_vector_transition_budget() -> None:
    learner = PPOLearner(_FakeAgent())
    collector = PPOCollector(_FakeVectorEnvironment(), learner)

    try:
        collector.collect(transition_budget=3)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("collector must reject a partial vector step")


def test_compiled_learner_compiles_only_policy_hot_paths(monkeypatch) -> None:
    compiled_functions = []

    def fake_compile(function, **kwargs):
        compiled_functions.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    learner = PPOLearner(_FakeAgent(), compile_policy=True)

    assert learner.compiled
    assert len(compiled_functions) == 2
    assert all(options == {"mode": "reduce-overhead"} for _, options in compiled_functions)


def test_flatten_rollout_data_keeps_time_before_environment_order() -> None:
    rollout_data = {
        "states": torch.arange(4).reshape(2, 2, 1, 1, 1),
        "actions": torch.tensor([[0, 1], [2, 3]]),
    }

    flattened = flatten_rollout_data(rollout_data)

    torch.testing.assert_close(flattened["states"].flatten(), torch.arange(4))
    torch.testing.assert_close(flattened["actions"], torch.arange(4))
