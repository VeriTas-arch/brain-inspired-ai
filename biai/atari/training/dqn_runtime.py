"""DQN collection with an explicit transition clock for scalar and batched environments."""

import torch


class DQNMetrics:
    """Keep every update's diagnostics, with one device read per bounded group."""

    def __init__(self, agent, callback, capacity=128):
        self.agent = getattr(agent, "agent", agent)
        self.ewc = self.agent is not agent
        self.callback = callback
        self.values = torch.empty((capacity, 3), device=self.agent.device)
        self.metadata = []

    def record(self, diagnostics, epsilon, regularized):
        self.values[len(self.metadata)].copy_(diagnostics)
        self.metadata.append((epsilon, regularized))
        if len(self.metadata) == len(self.values):
            self.flush()

    def flush(self):
        if not self.metadata:
            return
        rows = self.values[: len(self.metadata)].tolist()
        for row, (epsilon, regularized) in zip(rows, self.metadata, strict=True):
            metrics = self.agent.format_update_metrics(row, epsilon, regularized)
            if self.ewc and "regularization_loss" in metrics:
                metrics["ewc_loss"] = metrics.pop("regularization_loss")
            self.callback(metrics)
        self.metadata.clear()


def dqn_updates_due(start, count, *, learning_starts=10_000, frequency=4, post_increment=False):
    """Preserve each protocol's update count when several transitions arrive together."""
    lower, upper = start + int(post_increment), start + count + int(post_increment)
    first = max(lower, learning_starts)
    first += (-first) % frequency
    return len(range(first, upper, frequency))


class DQNCollector:
    """Store owned current/final observations; bootstrap through time limits only."""

    def __init__(self, environment, agent, replay, *, vectorized=True):
        self.environment, self.agent, self.replay = environment, agent, replay
        self.vectorized = vectorized
        self.num_envs = environment.num_envs if self.vectorized else 1
        self.states = environment.reset()
        self.episode_returns = torch.zeros(self.num_envs)
        self.episode_count = 0

    def collect(self):
        """Return completed episode rewards and one frame for optional recording."""
        if self.vectorized:
            states = self.states.clone()
            actions = (
                torch.tensor([self.agent.select_action(states[0])])
                if self.num_envs == 1
                else self.agent.select_actions(states)
            )
            step = self.environment.step_and_reset(actions)
            self.replay.add_batch(
                states, actions, step.rewards, step.transition_observations, step.terminated
            )
            self.states = step.observations
            rewards, done = step.rewards, step.terminated | step.truncated
            frame = states[0, 0]
        else:
            state = self.states
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated = self.environment.step(action)
            self.replay.add(state, action, reward, next_state, terminated)
            self.states = self.environment.reset() if terminated or truncated else next_state
            rewards, done = torch.tensor([reward]), torch.tensor([terminated or truncated])
            frame = state[0]
        self.episode_returns += rewards
        completed = self.episode_returns[done].tolist()
        self.episode_returns[done] = 0
        self.episode_count += len(completed)
        return completed, frame
