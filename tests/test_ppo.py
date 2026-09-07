"""Regression tests for PPO advantage estimation and updates."""

import math

import torch

from algorithms.ppo import (
    MultiHeadPPOAgent,
    PPOAgent,
    bootstrap_truncated_reward,
    generalized_advantage_estimate,
)


def test_rollout_sampling_matches_full_policy_evaluation() -> None:
    states = torch.randint(256, (3, 4, 84, 84), dtype=torch.uint8)
    agents = [
        PPOAgent(state_dim=4, action_dim=3, device="cpu"),
        MultiHeadPPOAgent(state_dim=4, device="cpu"),
    ]
    agents[1].register_task("task", 3)
    agents[1].set_task("task")

    for agent in agents:
        torch.manual_seed(19)
        expected_action, expected_log_prob, _, expected_value = agent.get_action_and_value(states)
        torch.manual_seed(19)
        action, log_prob, value = agent.sample_action_and_value(states)

        torch.testing.assert_close(action, expected_action)
        torch.testing.assert_close(log_prob, expected_log_prob)
        torch.testing.assert_close(value, expected_value)


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


def test_gae_keeps_vector_environments_independent() -> None:
    rewards = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        next_value=torch.zeros(2),
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


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


def test_ppo_update_flattens_vector_rollouts_after_gae() -> None:
    torch.manual_seed(13)
    agent = PPOAgent(state_dim=4, action_dim=2, device="cpu")
    states = torch.randint(256, (2, 2, 4, 84, 84), dtype=torch.uint8)
    flat_states = states.flatten(0, 1)
    with torch.no_grad():
        actions, log_probs, _, values = agent.get_action_and_value(flat_states)

    metrics = agent.update(
        {
            "states": states,
            "actions": actions.reshape(2, 2),
            "rewards": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "dones": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "log_probs": log_probs.reshape(2, 2),
            "values": values.reshape(2, 2),
        },
        next_value=torch.zeros(2),
        update_epochs=1,
        minibatch_size=2,
    )

    assert all(math.isfinite(value) for value in metrics.values())


def test_ppo_update_rejects_invalid_optimization_sizes() -> None:
    agent = PPOAgent(state_dim=4, action_dim=2, device="cpu")
    rollout = {
        "states": torch.zeros((1, 4, 84, 84), dtype=torch.uint8),
        "actions": torch.zeros(1, dtype=torch.long),
        "rewards": torch.zeros(1),
        "dones": torch.ones(1),
        "log_probs": torch.zeros(1),
        "values": torch.zeros(1),
    }

    for update_epochs, minibatch_size in ((0, 1), (1, 0)):
        try:
            agent.update(
                rollout,
                next_value=torch.zeros(1),
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid PPO optimization sizes must be rejected")


def test_ppo_metrics_average_all_minibatches() -> None:
    torch.manual_seed(11)
    agent = PPOAgent(state_dim=4, action_dim=2, lr=0.0, device="cpu")
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
        next_value=torch.zeros(1),
        update_epochs=1,
        minibatch_size=2,
    )

    assert abs(metrics["policy_loss"]) < 1e-6
