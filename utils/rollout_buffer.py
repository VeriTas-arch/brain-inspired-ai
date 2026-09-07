"""Rollout buffer for on-policy algorithms like PPO."""

import torch


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms like PPO."""

    def __init__(self, capacity: int = 128) -> None:
        """
        Initialize rollout buffer.

        Args:
            capacity: Number of steps to collect before updating (rollout length)
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.states: torch.Tensor | None = None
        self.actions = torch.empty(capacity, dtype=torch.long)
        self.rewards = torch.empty(capacity, dtype=torch.float32)
        self.dones = torch.empty(capacity, dtype=torch.float32)
        self.log_probs = torch.empty(capacity, dtype=torch.float32)
        self.values = torch.empty(capacity, dtype=torch.float32)
        self.pos = 0

    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Add a transition to the buffer."""
        if self.is_full():
            raise RuntimeError("Cannot add to a full rollout buffer")

        state = state.detach().to("cpu")
        if self.states is None:
            self.states = torch.empty(
                (self.capacity, *state.shape),
                dtype=state.dtype,
            )
        elif state.shape != self.states.shape[1:] or state.dtype != self.states.dtype:
            raise ValueError("Rollout states must have a consistent shape and dtype")

        self.states[self.pos].copy_(state)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.pos += 1

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.pos >= self.capacity

    def get_batch(self) -> dict[str, torch.Tensor]:
        """Get all collected data as a batch."""
        if self.states is None or self.pos == 0:
            raise ValueError("Cannot get a batch from an empty rollout buffer")
        batch_slice = slice(0, self.pos)
        return {
            "states": self.states[batch_slice],
            "actions": self.actions[batch_slice],
            "rewards": self.rewards[batch_slice],
            "dones": self.dones[batch_slice],
            "log_probs": self.log_probs[batch_slice],
            "values": self.values[batch_slice],
        }

    def ready_for_update(self, *, final: bool = False) -> bool:
        """Return whether a full or final partial rollout should be optimized."""
        return self.is_full() or (final and self.pos > 0)

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.pos
