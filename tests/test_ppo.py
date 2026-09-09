"""Regression tests for PPO advantage estimation and updates."""

import math

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from algorithms.ppo import (
    MultiHeadPPOAgent,
    PPOAgent,
    bootstrap_truncated_reward,
    generalized_advantage_estimate,
    optimize_ppo,
    ppo_minibatch_loss,
)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("multi_head", (False, True))
def test_policy_sampling_and_gradients_match_validated_distribution(device, multi_head) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if multi_head:
        agent = MultiHeadPPOAgent(4, device=device)
        agent.register_task("pong", 3)
        agent.set_task("pong")
        network, actor, critic = agent.backbone, agent.actors["pong"], agent.critics["pong"]
    else:
        agent = PPOAgent(4, 3, device=device)
        network, actor, critic = agent.network, agent.actor, agent.critic
    states = torch.randint(256, (3, 4, 84, 84), dtype=torch.uint8, device=device)
    hidden = network(states / 255.0)
    reference = Categorical(logits=actor(hidden), validate_args=True)
    expected_value = critic(hidden)
    torch.manual_seed(31)
    expected_action = reference.sample()
    expected_log_prob = reference.log_prob(expected_action)
    parameters = [*network.parameters(), *actor.parameters(), *critic.parameters()]
    expected_gradients = torch.autograd.grad(
        expected_log_prob.mean() + reference.entropy().mean() + expected_value.mean(), parameters
    )

    torch.manual_seed(31)
    action, log_prob, value = agent.sample_action_and_value(states)
    torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
    torch.testing.assert_close(log_prob, expected_log_prob, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
    _, log_prob, entropy, value = agent.get_action_and_value(states, action)
    gradients = torch.autograd.grad(log_prob.mean() + entropy.mean() + value.mean(), parameters)
    for expected, actual in zip(expected_gradients, gradients, strict=True):
        # Parallel convolution reductions can differ at float32 rounding precision.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_cpu_and_device_rollouts_produce_matching_ppo_updates(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(23)
    agent = PPOAgent(4, 2, device=device)
    reference = PPOAgent(4, 2, device=device)
    reference.load_checkpoint_state(agent.checkpoint_state())
    states = torch.randint(256, (4, 2, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        actions, log_probs, values = agent.sample_action_and_value(states.flatten(0, 1).to(device))
    rollout = {
        "states": states,
        "actions": actions.reshape(4, 2).cpu(),
        "log_probs": log_probs.reshape(4, 2),
        "values": values.reshape(4, 2),
        "rewards": torch.randn(4, 2),
        "dones": torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    }
    next_value = torch.tensor([0.3, -0.2], device=device)
    torch.manual_seed(37)
    actual = agent.update(rollout, next_value, update_epochs=2, minibatch_size=4)
    torch.manual_seed(37)
    expected = reference.update(
        {name: value.to(device) for name, value in rollout.items()},
        next_value,
        update_epochs=2,
        minibatch_size=4,
    )
    for name in actual:
        assert actual[name] == pytest.approx(expected[name], abs=1e-5, rel=1e-4)
    for module_name in ("network", "actor", "critic"):
        for left, right in zip(
            getattr(agent, module_name).parameters(),
            getattr(reference, module_name).parameters(),
            strict=True,
        ):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)


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


def test_multi_head_ppo_uses_the_shared_update_path() -> None:
    torch.manual_seed(8)
    agent = MultiHeadPPOAgent(state_dim=4, device="cpu")
    agent.register_task("task", 2)
    agent.set_task("task")
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

    diagnostics = iter(((1.0, 2.0, 3.0, 0.1, 0.2), (3.0, 4.0, 5.0, 0.3, 0.4)))

    def minibatch_loss(*args):
        loss = agent.actor.weight.sum() * 0
        return (loss, *(torch.tensor(value) for value in next(diagnostics)))

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
        minibatch_loss=minibatch_loss,
    )

    assert metrics == pytest.approx(
        {"policy_loss": 2.0, "value_loss": 3.0, "entropy": 4.0, "approx_kl": 0.2, "clipfrac": 0.3}
    )


def test_partial_final_minibatch_uses_eager_loss_shape() -> None:
    policy = nn.Linear(2, 3)
    critic = nn.Linear(2, 1)
    parameters = [*policy.parameters(), *critic.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0.0)
    states = torch.randn(3, 2)
    actions = torch.tensor([0, 1, 2])

    def evaluate(input_states, input_actions=None):
        distribution = Categorical(logits=policy(input_states))
        if input_actions is None:
            input_actions = distribution.sample()
        return (
            input_actions,
            distribution.log_prob(input_actions),
            distribution.entropy(),
            critic(input_states),
        )

    with torch.no_grad():
        _, log_probs, _, values = evaluate(states, actions)

    compiled_batch_sizes = []

    def compiled_loss(*arguments):
        compiled_batch_sizes.append(len(arguments[0]))
        return ppo_minibatch_loss(
            *arguments,
            policy_evaluator=evaluate,
            clip_coef=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
        )

    optimize_ppo(
        {
            "states": states,
            "actions": actions,
            "rewards": torch.zeros(3),
            "dones": torch.tensor([0.0, 0.0, 1.0]),
            "log_probs": log_probs,
            "values": values.flatten(),
        },
        next_value=torch.zeros(1),
        device=torch.device("cpu"),
        optimizer=optimizer,
        parameters=parameters,
        policy_evaluator=evaluate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=1,
        minibatch_size=2,
        minibatch_loss=compiled_loss,
    )

    assert compiled_batch_sizes == [2]
