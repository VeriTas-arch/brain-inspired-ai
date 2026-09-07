"""Regression tests for PPO advantage estimation and updates."""

import math

import torch

from algorithms.ppo import PPOAgent, bootstrap_truncated_reward, generalized_advantage_estimate


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


def test_time_limit_reward_bootstraps_without_crossing_the_reset_boundary() -> None:
    reward = bootstrap_truncated_reward(
        1.0,
        2.0,
        terminated=False,
        truncated=True,
        gamma=0.9,
    )
    terminal_reward = bootstrap_truncated_reward(
        1.0,
        2.0,
        terminated=True,
        truncated=True,
        gamma=0.9,
    )

    assert reward == 2.8
    assert terminal_reward == 1.0


def test_ppo_update_accepts_uint8_rollout_states() -> None:
    torch.manual_seed(7)
    agent = PPOAgent(state_dim=4, action_dim=2, device="cpu")
    states = torch.randint(256, (4, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        actions, log_probs, _, values = agent.get_action_and_value(states)

    metrics = agent.update(
        {
            "states": states,
            "actions": actions,
            "rewards": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "dones": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "log_probs": log_probs,
            "values": values.flatten(),
        },
        next_value=torch.tensor([0.0]),
        update_epochs=1,
        minibatch_size=2,
    )

    assert all(math.isfinite(value) for value in metrics.values())
