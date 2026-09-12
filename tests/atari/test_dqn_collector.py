"""Replay ownership, time-limit bootstrapping, and DQN update cadence."""

import pytest
import torch

from biai.atari.environments import VectorStep
from biai.atari.training import DQNCollector, ReplayBuffer, dqn_updates_due


@pytest.mark.parametrize("post_increment", (False, True))
@pytest.mark.parametrize("num_envs", (1, 2, 4, 8, 16))
def test_batched_update_clock_matches_scalar_protocol(num_envs, post_increment):
    expected = sum(
        step >= 10_000 and step % 4 == 0
        for step in range(int(post_increment), 10_032 + int(post_increment))
    )
    actual = sum(
        dqn_updates_due(step, num_envs, post_increment=post_increment)
        for step in range(0, 10_032, num_envs)
    )
    assert actual == expected


def test_vector_dqn_preserves_shared_current_and_terminal_observations():
    class Environment:
        num_envs = 2
        states = torch.ones(2, 4, 1, 1, dtype=torch.uint8)

        def reset(self):
            return self.states

        def step_and_reset(self, actions):
            self.states.fill_(99)
            return VectorStep(
                self.states,
                torch.full_like(self.states, 7),
                torch.tensor([3.0, 5.0]),
                torch.tensor([True, False]),
                torch.tensor([False, True]),
            )

    class Agent:
        def select_actions(self, states):
            return torch.tensor([0, 1])

    replay = ReplayBuffer(8)
    collector = DQNCollector(Environment(), Agent(), replay)
    rewards, _ = collector.collect()
    batch = replay.sample(2)
    order = batch["actions"].argsort()
    assert rewards == [3.0, 5.0] and collector.episode_count == 2
    assert (batch["states"] == 1).all() and (batch["next_states"] == 7).all()
    torch.testing.assert_close(batch["dones"][order], torch.tensor([1.0, 0.0]))
