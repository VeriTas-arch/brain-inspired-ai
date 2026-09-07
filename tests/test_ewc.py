"""Regression tests for EWC Fisher estimation and persistence."""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms import BaseAgent, EWCWrapper


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
        penalty = regularizer() if regularizer is not None else torch.zeros(())
        self.optimizer.zero_grad()
        (loss + penalty).backward()
        self.optimizer.step()
        return {"loss": loss.item(), "regularization_loss": penalty.detach().item()}

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


def test_fisher_averages_per_sample_squared_gradients() -> None:
    agent = TinyDQN()
    with torch.no_grad():
        agent.network.weight.zero_()
        agent.network.bias.zero_()
        agent.target_network.load_state_dict(agent.network.state_dict())

    fisher = EWCWrapper(agent).compute_fisher_information(dqn_batch(), num_samples=2)

    torch.testing.assert_close(fisher["network.weight"][0, 0], torch.tensor(4.0))
    torch.testing.assert_close(fisher["network.bias"][0], torch.tensor(4.0))


def test_ppo_fisher_does_not_require_dqn_transition_fields() -> None:
    wrapper = EWCWrapper(TinyPPO())
    fisher = wrapper.compute_fisher_information(
        {"states": torch.ones(2, 2), "actions": torch.tensor([0, 1])},
        num_samples=2,
    )

    assert "network.weight" in fisher
    assert "actor.weight" in fisher


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
    for name, expected in wrapper.task_fisher[0].items():
        torch.testing.assert_close(restored.task_fisher[0][name], expected)
