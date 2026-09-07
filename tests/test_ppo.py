"""Regression tests for PPO advantage estimation."""

import torch

from algorithms.ppo import generalized_advantage_estimate


def test_gae_stops_at_transition_terminal() -> None:
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([1.0, 2.0, 0.0])
    dones = torch.tensor([0.0, 1.0, 0.0])

    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        next_value=torch.tensor([0.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([-1.0, -2.0, 1.0]))
    torch.testing.assert_close(returns, torch.tensor([0.0, 0.0, 1.0]))
