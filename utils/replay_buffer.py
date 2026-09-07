"""Experience replay buffer."""

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Experience replay buffer for RL."""

    def __init__(self, capacity: int = 100000) -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append(
            {
                "state": state.detach().cpu().numpy().copy(),
                "action": action,
                "reward": reward,
                "next_state": next_state.detach().cpu().numpy().copy(),
                "done": done,
            }
        )

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.buffer:
            raise ValueError("Cannot sample from an empty replay buffer")

        actual_batch_size = min(batch_size, len(self.buffer))
        choice = np.random.choice if rng is None else rng.choice
        indices = choice(len(self.buffer), actual_batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.from_numpy(np.stack([b["state"] for b in batch]))
        actions = torch.from_numpy(np.array([b["action"] for b in batch])).long()
        rewards = torch.from_numpy(np.array([b["reward"] for b in batch])).float()
        next_states = torch.from_numpy(np.stack([b["next_state"] for b in batch]))
        dones = torch.from_numpy(np.array([b["done"] for b in batch])).float()

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return len(self.buffer) >= batch_size
