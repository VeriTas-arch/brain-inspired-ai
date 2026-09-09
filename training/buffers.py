"""Replay and rollout buffers for off-policy and on-policy training."""

from collections.abc import Sequence

import numpy as np
import torch


class ReplayBuffer:
    """Uniform replay backed by contiguous CPU arrays and an independent RNG."""

    def __init__(self, capacity: int = 100000, *, seed: int | Sequence[int] = 0) -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            seed: Independent replay RNG seed; sampling never advances exploration/task RNGs
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._storage: dict[str, np.ndarray] = {}
        self._length = 0
        self._next_index = 0
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        transition = {
            "states": state.detach().cpu().numpy(),
            "actions": np.int64(action),
            "rewards": np.float32(reward),
            "next_states": next_state.detach().cpu().numpy(),
            "dones": np.float32(done),
        }
        if not self._storage:
            self._storage = {
                key: np.empty((self.capacity, *np.shape(value)), dtype=np.asarray(value).dtype)
                for key, value in transition.items()
            }
        for key, value in transition.items():
            self._storage[key][self._next_index] = value
        self._next_index = (self._next_index + 1) % self.capacity
        self._length = min(self.capacity, self._length + 1)

    def add_batch(self, states, actions, rewards, next_states, dones) -> None:
        """Copy simultaneous transitions before an environment reuses shared memory."""
        columns = {
            "states": states.detach().cpu().numpy(),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": next_states.detach().cpu().numpy(),
            "dones": np.asarray(dones, dtype=np.float32),
        }
        count = len(states)
        if not count or any(len(value) != count for value in columns.values()):
            raise ValueError("Replay columns must have the same nonempty leading dimension")
        if not self._storage:
            self._storage = {
                key: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for key, value in columns.items()
            }
        kept = min(count, self.capacity)
        start = (self._next_index + count - kept) % self.capacity
        first = min(kept, self.capacity - start)
        for key, value in columns.items():
            tail = value[-kept:]
            self._storage[key][start : start + first] = tail[:first]
            self._storage[key][: kept - first] = tail[first:]
        self._next_index = (self._next_index + count) % self.capacity
        self._length = min(self.capacity, self._length + count)

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._length:
            raise ValueError("Cannot sample from an empty replay buffer")
        actual_batch_size = min(batch_size, self._length)
        choice = self._rng.choice if rng is None else rng.choice
        indices = choice(self._length, actual_batch_size, replace=False)
        # Logical indices run oldest to newest, including after ring wraparound.
        if self._length == self.capacity:
            indices = (indices + self._next_index) % self.capacity
        return {
            key: torch.from_numpy(np.take(array, indices, axis=0))
            for key, array in self._storage.items()
        }

    def __len__(self) -> int:
        """Return buffer size."""
        return self._length

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self._length >= batch_size


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms like PPO."""

    def __init__(
        self, capacity: int = 128, num_envs: int = 1, policy_device: torch.device | str = "cpu"
    ) -> None:
        """
        Initialize rollout buffer.

        Args:
            capacity: Number of time steps to collect before updating
            num_envs: Number of independent environments collected at each time step
            policy_device: Store log probabilities and values beside the policy to avoid
                copying them to the CPU at every environment step
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
        self.log_probs = torch.empty(scalar_shape, dtype=torch.float32, device=policy_device)
        self.values = torch.empty(scalar_shape, dtype=torch.float32, device=policy_device)
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

        state = state.detach()
        if self.num_envs > 1 and (state.ndim == 0 or state.shape[0] != self.num_envs):
            raise ValueError(f"Expected states from {self.num_envs} environments")
        if self.states is None:
            self.states = torch.empty(
                (self.capacity, *state.shape),
                dtype=state.dtype,
                device=state.device,
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
