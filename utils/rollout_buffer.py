"""Rollout buffer for on-policy algorithms like PPO."""

import torch


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
