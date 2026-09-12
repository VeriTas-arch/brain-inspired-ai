"""Regression tests for EWC importance estimation and persistence."""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from biai.atari.algorithms import BaseAgent, EWCWrapper, summarize_parameter_importance


class TinyDQN(BaseAgent):
    """Small DQN-shaped agent used to make Fisher tests exact and fast."""

    def __init__(self) -> None:
        super().__init__(state_dim=2, action_dim=2, device="cpu")
        self.network = nn.Linear(2, 2)
        self.target_network = deepcopy(self.network)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.1)
        self.gamma = 0.0

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        del deterministic
        return self.network(state).argmax().item()

    def update(self, batch, regularizer=None) -> dict[str, float]:
        predictions = self.network(batch["states"]).gather(1, batch["actions"][:, None]).flatten()
        loss = F.mse_loss(predictions, batch["rewards"])
        penalty = regularizer() if regularizer is not None else None
        self.optimizer.zero_grad()
        (loss if penalty is None else loss + penalty).backward()
        self.optimizer.step()
        metrics = {"loss": loss.item()}
        if penalty is not None:
            metrics["regularization_loss"] = penalty.detach().item()
        return metrics

    def checkpoint_state(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class TinyPPO(BaseAgent):
    """Small PPO-shaped agent used to test EWC dispatch."""

    def __init__(self) -> None:
        super().__init__(state_dim=2, action_dim=2, device="cpu")
        self.network = nn.Linear(2, 3)
        self.actor = nn.Linear(3, 2)
        self.critic = nn.Linear(3, 1)
        self.optimizer = torch.optim.SGD(
            [*self.network.parameters(), *self.actor.parameters(), *self.critic.parameters()],
            lr=0.1,
        )

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        del deterministic
        return self.actor(self.network(state / 255.0)).argmax().item()

    def update(self, batch) -> dict[str, float]:
        del batch
        return {}


def dqn_batch() -> dict[str, torch.Tensor]:
    return {
        "states": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "actions": torch.tensor([0, 0]),
        "rewards": torch.tensor([1.0, -1.0]),
        "next_states": torch.zeros(2, 2),
        "dones": torch.ones(2),
    }


def test_importance_averages_per_sample_squared_gradients() -> None:
    agent = TinyDQN()
    with torch.no_grad():
        agent.network.weight.zero_()
        agent.network.bias.zero_()
        agent.target_network.load_state_dict(agent.network.state_dict())

    fisher = EWCWrapper(agent).compute_parameter_importance(dqn_batch(), num_samples=2)

    torch.testing.assert_close(fisher["network.weight"][0, 0], torch.tensor(4.0))
    torch.testing.assert_close(fisher["network.bias"][0], torch.tensor(4.0))


def test_importance_sampling_does_not_advance_training_rng() -> None:
    agent = TinyDQN()
    batch = {
        name: values.repeat((60,) + (1,) * (values.ndim - 1))
        for name, values in dqn_batch().items()
    }

    torch.manual_seed(17)
    expected = torch.rand(3)
    torch.manual_seed(17)
    EWCWrapper(agent).compute_parameter_importance(batch, num_samples=2)

    torch.testing.assert_close(torch.rand(3), expected)


def test_ppo_fisher_does_not_require_dqn_transition_fields() -> None:
    wrapper = EWCWrapper(TinyPPO())
    fisher = wrapper.compute_parameter_importance(
        {"states": torch.ones(2, 2), "actions": torch.tensor([0, 1])},
        num_samples=2,
    )

    assert "network.weight" in fisher
    assert "actor.weight" in fisher

    summary = summarize_parameter_importance(fisher)
    assert summary["network"]["nonzero_fraction"] > 0
    assert summary["actor"]["nonzero_fraction"] > 0
    assert "critic" not in summary


def test_dqn_and_ppo_importance_estimators_are_named_distinctly() -> None:
    dqn_diagnostic = EWCWrapper(TinyDQN()).consolidate_weights(dqn_batch())
    ppo_diagnostic = EWCWrapper(TinyPPO()).consolidate_weights(
        {"states": torch.ones(2, 2), "actions": torch.tensor([0, 1])}
    )

    assert dqn_diagnostic["estimator"] == "td_mse_squared_gradient_importance"
    assert not dqn_diagnostic["is_empirical_fisher"]
    assert ppo_diagnostic["estimator"] == "policy_nll_empirical_fisher"
    assert ppo_diagnostic["is_empirical_fisher"]
    assert ppo_diagnostic["protected_modules"] == ["network", "actor"]


def test_ewc_checkpoint_restores_consolidated_state(tmp_path) -> None:
    wrapper = EWCWrapper(TinyDQN(), ewc_lambda=1.5)
    wrapper.consolidate_weights(dqn_batch())
    checkpoint = tmp_path / "ewc.pt"
    wrapper.save(str(checkpoint))

    restored = EWCWrapper(TinyDQN())
    restored.load(str(checkpoint))

    assert restored.current_task_id == 1
    assert restored.ewc_lambda == 1.5
    assert restored.task_weights.keys() == wrapper.task_weights.keys()
    assert restored.task_fisher_summary == wrapper.task_fisher_summary
    for name, expected in wrapper.task_fisher[0].items():
        torch.testing.assert_close(restored.task_fisher[0][name], expected)
    for name, expected in wrapper.aggregated_fisher.items():
        torch.testing.assert_close(restored.aggregated_fisher[name], expected)


def test_aggregated_ewc_penalty_matches_explicit_task_sum(monkeypatch) -> None:
    agent = TinyDQN()
    wrapper = EWCWrapper(agent, ewc_lambda=2.0)
    fishers = iter(
        (
            {
                "network.weight": torch.full_like(agent.network.weight, 2.0),
                "network.bias": torch.full_like(agent.network.bias, 2.0),
            },
            {
                "network.weight": torch.full_like(agent.network.weight, 4.0),
                "network.bias": torch.full_like(agent.network.bias, 4.0),
            },
        )
    )
    monkeypatch.setattr(wrapper, "compute_parameter_importance", lambda batch: next(fishers))

    with torch.no_grad():
        agent.network.weight.fill_(1.0)
        agent.network.bias.fill_(1.0)
    wrapper.consolidate_weights(dqn_batch())
    with torch.no_grad():
        agent.network.weight.fill_(3.0)
        agent.network.bias.fill_(3.0)
    wrapper.consolidate_weights(dqn_batch())
    with torch.no_grad():
        agent.network.weight.fill_(5.0)
        agent.network.bias.fill_(5.0)

    explicit_penalty = torch.zeros(())
    for task_id, weights in wrapper.task_weights.items():
        for name, parameter in wrapper._collect_regularized_params().items():
            explicit_penalty += (
                wrapper.task_fisher[task_id][name] * (parameter - weights[name]).square()
            ).sum()

    torch.testing.assert_close(wrapper.compute_ewc_loss(), explicit_penalty)


def test_runtime_regularizer_matches_dynamic_penalty() -> None:
    wrapper = EWCWrapper(TinyDQN(), ewc_lambda=1.5)
    wrapper.consolidate_weights(dqn_batch())
    with torch.no_grad():
        wrapper.agent.network.weight.add_(0.25)

    wrapper.configure_regularizer()
    torch.testing.assert_close(
        wrapper._runtime_regularizer(),
        wrapper.compute_ewc_loss(),
    )


def test_first_task_update_skips_empty_regularizer() -> None:
    wrapper = EWCWrapper(TinyDQN())

    metrics = wrapper.update(dqn_batch())

    assert "ewc_loss" not in metrics


def test_legacy_ewc_checkpoint_rebuilds_aggregates(tmp_path) -> None:
    source = EWCWrapper(TinyDQN(), ewc_lambda=1.5)
    source.consolidate_weights(dqn_batch())
    checkpoint = tmp_path / "ewc-v1.pt"
    torch.save(
        {
            "format": "ewc-v1",
            "agent": source.agent.checkpoint_state(),
            "ewc_lambda": source.ewc_lambda,
            "task_weights": source.task_weights,
            "task_fisher": source.task_fisher,
            "task_fisher_summary": source.task_fisher_summary,
            "current_task_id": source.current_task_id,
        },
        checkpoint,
    )

    restored = EWCWrapper(TinyDQN())
    restored.load(str(checkpoint))

    assert restored.aggregated_fisher.keys() == source.aggregated_fisher.keys()
    for name, expected in source.aggregated_fisher.items():
        torch.testing.assert_close(restored.aggregated_fisher[name], expected)
