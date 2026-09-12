"""Agent actions and checkpoint round trips."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from biai.atari.algorithms import DQNAgent, MultiHeadDQNAgent, MultiHeadPPOAgent, PPOAgent


@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
def test_compiled_greedy_batches_match_scalar_actions_without_consuming_rng(
    algorithm, fresh_compiler_state
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agent = (MultiHeadDQNAgent if algorithm == "dqn" else MultiHeadPPOAgent)(4, device="cuda")
    for task, actions in (("old", 6), ("new", 4)):
        agent.register_task(task, actions)
    states = torch.randint(256, (8, 4, 84, 84), dtype=torch.uint8)
    expected = {}
    for task in ("old", "new"):
        agent.set_task(task)
        expected[task] = [agent.select_action(state, deterministic=True) for state in states]
    agent.configure_runtime(compile_enabled=True)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    for task in ("old", "new", "old"):
        agent.set_task(task)
        actions = (
            agent.select_actions(states, deterministic=True)
            if algorithm == "dqn"
            else agent.select_actions(states)
        )
        assert actions.tolist() == expected[task]
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    for first, second in zip(np.random.get_state(), numpy_rng, strict=True):
        np.testing.assert_equal(first, second)


@pytest.mark.parametrize("agent_type", (DQNAgent, PPOAgent, MultiHeadDQNAgent, MultiHeadPPOAgent))
def test_checkpoint_restores_native_observation_protocol(agent_type, tmp_path):
    options = {"action_dim": 2} if agent_type in (DQNAgent, PPOAgent) else {}
    agent = agent_type(4, device="cpu", **options)
    if hasattr(agent, "register_task"):
        agent.register_task("pong", 2)
        agent.set_task("pong")
    agent.environment_protocol = "ale_native_v1"
    path = tmp_path / "model.pt"
    agent.save(str(path))
    restored = agent_type(4, device="cpu", **options)
    restored.load(str(path))
    assert restored.environment_protocol == "ale_native_v1"


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


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_multihead_dqn_checkpoint_restores_heads_and_optimizer(tmp_path, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = MultiHeadDQNAgent(state_dim=4, device=device)
    source.register_task("pong", np.int64(2))
    source.register_task("breakout", np.int64(3))
    source.set_task("breakout")

    batch = {
        "states": torch.randint(256, (2, 4, 84, 84), dtype=torch.uint8).float(),
        "actions": torch.tensor([0, 2]),
        "rewards": torch.tensor([1.0, -1.0]),
        "next_states": torch.randint(256, (2, 4, 84, 84), dtype=torch.uint8).float(),
        "dones": torch.tensor([0.0, 1.0]),
    }
    source.update(batch)
    assert source.task_epsilons["breakout"] < source.task_epsilons["pong"]

    checkpoint = tmp_path / "multihead-dqn.pt"
    source.save(str(checkpoint))

    restored = MultiHeadDQNAgent(state_dim=4, device=device)
    restored.load(str(checkpoint))

    assert list(restored.heads) == ["pong", "breakout"]
    assert restored.current_task == "breakout"
    assert restored.task_epsilons == source.task_epsilons
    assert len(restored.optimizer.param_groups) == len(source.optimizer.param_groups)
    for task_id in source.heads:
        for expected, actual in zip(
            source.heads[task_id].parameters(), restored.heads[task_id].parameters()
        ):
            torch.testing.assert_close(actual, expected)
    # Continue from inherited Adam moments after loading the checkpoint.
    source.update(batch)
    restored.update(batch)
    for expected, actual in zip(source.backbone.parameters(), restored.backbone.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_multihead_ppo_checkpoint_restores_task_heads(tmp_path, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = MultiHeadPPOAgent(state_dim=4, device=device)
    source.register_task("pong", np.int64(2))
    source.register_task("breakout", np.int64(3))
    source.set_task("pong")
    with torch.no_grad():
        source.actors["pong"].bias.fill_(0.25)
        source.critics["breakout"].bias.fill_(-0.5)

    checkpoint = tmp_path / "multihead-ppo.pt"
    source.save(str(checkpoint))

    restored = MultiHeadPPOAgent(state_dim=4, device=device)
    restored.load(str(checkpoint))

    assert list(restored.actors) == ["pong", "breakout"]
    assert restored.current_task == "pong"
    torch.testing.assert_close(restored.actors["pong"].bias, source.actors["pong"].bias)
    torch.testing.assert_close(restored.critics["breakout"].bias, source.critics["breakout"].bias)
