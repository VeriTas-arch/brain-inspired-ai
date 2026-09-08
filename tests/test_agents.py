"""Agent actions and checkpoint round trips."""

import torch
import torch.nn as nn

from algorithms import DQNAgent, MultiHeadDQNAgent, MultiHeadPPOAgent, PPOAgent


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


def test_multihead_dqn_checkpoint_restores_heads_and_optimizer(tmp_path) -> None:
    source = MultiHeadDQNAgent(state_dim=4, device="cpu")
    source.register_task("pong", 2)
    source.register_task("breakout", 3)
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

    restored = MultiHeadDQNAgent(state_dim=4, device="cpu")
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


def test_multihead_ppo_checkpoint_restores_task_heads(tmp_path) -> None:
    source = MultiHeadPPOAgent(state_dim=4, device="cpu")
    source.register_task("pong", 2)
    source.register_task("breakout", 3)
    source.set_task("pong")
    with torch.no_grad():
        source.actors["pong"].bias.fill_(0.25)
        source.critics["breakout"].bias.fill_(-0.5)

    checkpoint = tmp_path / "multihead-ppo.pt"
    source.save(str(checkpoint))

    restored = MultiHeadPPOAgent(state_dim=4, device="cpu")
    restored.load(str(checkpoint))

    assert list(restored.actors) == ["pong", "breakout"]
    assert restored.current_task == "pong"
    torch.testing.assert_close(restored.actors["pong"].bias, source.actors["pong"].bias)
    torch.testing.assert_close(restored.critics["breakout"].bias, source.critics["breakout"].bias)
