"""Replay and rollout buffers for off-policy and on-policy training."""

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


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms like PPO."""

    def __init__(self, capacity: int = 128, num_envs: int = 1) -> None:
        """
        Initialize rollout buffer.

        Args:
            capacity: Number of time steps to collect before updating
            num_envs: Number of independent environments collected at each time step
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.capacity = capacity
        self.num_envs = num_envs
        self.states: torch.Tensor | None = None
        scalar_shape = (capacity,) if num_envs == 1 else (capacity, num_envs)
        self.actions = torch.empty(scalar_shape, dtype=torch.long)
        self.rewards = torch.empty(scalar_shape, dtype=torch.float32)
        self.dones = torch.empty(scalar_shape, dtype=torch.float32)
        self.log_probs = torch.empty(scalar_shape, dtype=torch.float32)
        self.values = torch.empty(scalar_shape, dtype=torch.float32)
        self.pos = 0
        self._pending_step = False

    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self._pending_step = False

    def add(
        self,
        state: torch.Tensor,
        action: int | torch.Tensor,
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
        log_prob: float | torch.Tensor,
        value: float | torch.Tensor,
    ) -> None:
        """Add a transition to the buffer."""
        self.start_step(state, action, log_prob, value)
        self.finish_step(reward, done)

    def start_step(
        self,
        state: torch.Tensor,
        action: int | torch.Tensor,
        log_prob: float | torch.Tensor,
        value: float | torch.Tensor,
    ) -> None:
        """Store policy outputs before a shared observation buffer is overwritten."""
        if self.is_full():
            raise RuntimeError("Cannot add to a full rollout buffer")
        if self._pending_step:
            raise RuntimeError("Previous rollout step has not been completed")

        state = state.detach().to("cpu")
        if self.num_envs > 1 and (state.ndim == 0 or state.shape[0] != self.num_envs):
            raise ValueError(f"Expected states from {self.num_envs} environments")
        if self.states is None:
            self.states = torch.empty(
                (self.capacity, *state.shape),
                dtype=state.dtype,
            )
        elif state.shape != self.states.shape[1:] or state.dtype != self.states.dtype:
            raise ValueError("Rollout states must have a consistent shape and dtype")

        self.states[self.pos].copy_(state)
        self.actions[self.pos].copy_(torch.as_tensor(action, dtype=torch.long))
        self.log_probs[self.pos].copy_(torch.as_tensor(log_prob, dtype=torch.float32))
        self.values[self.pos].copy_(torch.as_tensor(value, dtype=torch.float32))
        self._pending_step = True

    def finish_step(
        self,
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
    ) -> None:
        """Complete the pending step once the environment result is available."""
        if not self._pending_step:
            raise RuntimeError("No pending rollout step to complete")
        self.rewards[self.pos].copy_(torch.as_tensor(reward, dtype=torch.float32))
        self.dones[self.pos].copy_(torch.as_tensor(done, dtype=torch.float32))
        self.pos += 1
        self._pending_step = False

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.pos >= self.capacity

    def get_batch(self) -> dict[str, torch.Tensor]:
        """Get all collected data as a batch."""
        if self._pending_step:
            raise RuntimeError("Cannot read a rollout with an incomplete step")
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

    @property
    def transition_count(self) -> int:
        """Return the number of stored environment transitions."""
        return self.pos * self.num_envs
