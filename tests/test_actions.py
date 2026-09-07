"""Tests for deterministic evaluation policies."""

import torch
import torch.nn as nn

from algorithms import DQNAgent, PPOAgent


def test_dqn_deterministic_action_ignores_epsilon() -> None:
    agent = DQNAgent(
        state_dim=4,
        action_dim=3,
        epsilon_start=1.0,
        epsilon_end=1.0,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in agent.network.parameters():
            parameter.zero_()
        output_layer = agent.network.network[-1]
        assert isinstance(output_layer, nn.Linear)
        output_layer.bias.copy_(torch.tensor([0.0, 2.0, 1.0]))

    state = torch.zeros(4, 84, 84, dtype=torch.uint8)

    assert {agent.select_action(state, deterministic=True) for _ in range(20)} == {1}


def test_ppo_deterministic_action_uses_highest_logit() -> None:
    agent = PPOAgent(state_dim=4, action_dim=3, device="cpu")
    with torch.no_grad():
        for parameter in agent.network.parameters():
            parameter.zero_()
        agent.actor.weight.zero_()
        agent.actor.bias.copy_(torch.tensor([0.0, 1.0, 3.0]))

    state = torch.zeros(4, 84, 84, dtype=torch.uint8)

    assert agent.select_action(state, deterministic=True) == 2
