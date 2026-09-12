"""DQN (Deep Q-Network) implementation."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import AtariBackbone, BaseAgent, SimpleNet

DEFAULT_DQN_LEARNING_STARTS = 10_000


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Linear epsilon schedule."""
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def _clip_dqn_gradients(parameter_groups, max_norm):
    """Preserve the separate backbone/head clipping used by multi-task DQN."""
    for parameters in parameter_groups:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _DQNComputation:
    """Tensor-only inference and TD loss; modules keep their checkpoint identities."""

    def __init__(
        self,
        online,
        target,
        gamma,
        *,
        head=None,
        target_head=None,
        compiled=False,
        capture_updates=True,
    ):
        self.compiled = compiled
        self.capture_updates = capture_updates
        self._update_graph = None
        self._update_key = None

        def greedy(states):
            values = online(states)
            if head is not None:
                values = head(values)
            return values.argmax(dim=1)

        def loss(states, actions, rewards, next_states, dones, regularizer):
            values = online(states)
            if head is not None:
                values = head(values)
            selected = values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_values = target(next_states)
                if target_head is not None:
                    next_values = target_head(next_values)
                expected = rewards.flatten() + gamma * next_values.max(dim=1)[0] * (
                    1 - dones.flatten()
                )
            td_loss = F.mse_loss(expected, selected)
            penalty = td_loss.new_zeros(()) if regularizer is None else regularizer()
            return td_loss + penalty, torch.stack(
                (td_loss.detach(), values.detach().mean(), penalty.detach())
            )

        self.greedy = (
            torch.compile(greedy, mode="reduce-overhead", fullgraph=True) if compiled else greedy
        )
        self.loss = (
            torch.compile(loss, mode="reduce-overhead", fullgraph=True) if compiled else loss
        )
        self._eager_loss = loss

    def update(self, optimizer, groups, max_norm, batch, regularizer, clip_gradients):
        if self.compiled and self.capture_updates and batch["states"].is_cuda:
            from biai.atari.training.cuda_update import CudaUpdate

            inputs = tuple(
                batch[key] for key in ("states", "actions", "rewards", "next_states", "dones")
            )
            key = (
                tuple((value.shape, value.dtype) for value in inputs),
                regularizer,
                getattr(optimizer, "_subspace_projection", None),
            )
            if key != self._update_key:
                self._update_graph = CudaUpdate(
                    optimizer,
                    lambda *values: self._eager_loss(*values, regularizer),
                    lambda: _clip_dqn_gradients(groups, max_norm),
                    inputs,
                )
                self._update_key = key
            return self._update_graph(*inputs)
        self.begin_step()
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = self.loss(**batch, regularizer=regularizer)
        loss.backward()
        clip_gradients(groups, max_norm)
        optimizer.step()
        return diagnostics

    def begin_step(self):
        # Inference outputs are consumed immediately; a TD step finishes its backward and metrics
        # before another invocation. Declare that lifetime explicitly to CUDA Graph Trees.
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()


def epsilon_greedy_batch(compute, states, epsilon, action_dim, device, deterministic=False):
    """Use one fixed-shape inference batch and independent exploration per environment."""
    random_mask = (
        np.zeros(len(states), dtype=np.bool_)
        if deterministic
        else np.random.random(len(states)) < epsilon
    )
    random_actions = np.random.randint(action_dim, size=int(random_mask.sum()))
    if random_mask.all():
        return torch.from_numpy(random_actions)
    with torch.inference_mode():
        compute.begin_step()
        actions = compute.greedy(states.to(device, non_blocking=True)).cpu()
        actions[torch.from_numpy(random_mask)] = torch.from_numpy(random_actions)
    return actions


class DQNAgent(BaseAgent):
    """DQN Agent for Atari games."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_fraction: float = 0.10,
        total_timesteps: int = 10000000,
        target_update_freq: int = 1000,
        tau: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, device)

        self.network = SimpleNet(state_dim, action_dim).to(self.device)
        self.target_network = SimpleNet(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, fused=self.device.type == "cuda"
        )

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_fraction = epsilon_fraction
        self.total_timesteps = total_timesteps
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.update_count = 0
        self.global_step = 0
        self.q_value_history = []
        self.configure_runtime(compile_enabled=False)

    def configure_runtime(
        self, *, compile_enabled: bool = False, capture_updates: bool = True
    ) -> None:
        """Select eager or compiled tensor computation without wrapping saved modules."""
        self._compute = _DQNComputation(
            self.network,
            self.target_network,
            self.gamma,
            compiled=compile_enabled,
            capture_updates=capture_updates,
        )
        self._gradient_groups = (tuple(self.network.parameters()),)
        self._clip_gradients = (
            torch.compile(_clip_dqn_gradients, mode="reduce-overhead", fullgraph=True)
            if compile_enabled
            else _clip_dqn_gradients
        )

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        epsilon = linear_schedule(
            self.epsilon_start,
            self.epsilon_end,
            int(self.epsilon_fraction * self.total_timesteps),
            self.global_step,
        )

        if not deterministic and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.inference_mode():
            self._compute.begin_step()
            state = state.unsqueeze(0).to(self.device, non_blocking=True)
            return self._compute.greedy(state).item()

    def select_actions(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        epsilon = linear_schedule(
            self.epsilon_start,
            self.epsilon_end,
            int(self.epsilon_fraction * self.total_timesteps),
            self.global_step,
        )
        return epsilon_greedy_batch(
            self._compute, states, epsilon, self.action_dim, self.device, deterministic
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        regularizer: Callable[[], torch.Tensor] | None = None,
        *,
        metrics_sink: Callable | None = None,
    ) -> dict[str, float]:
        """Update DQN with a batch of experiences."""
        device_batch = {
            name: value.to(self.device, non_blocking=True) for name, value in batch.items()
        }
        diagnostics = self._compute.update(
            self.optimizer,
            self._gradient_groups,
            10.0,
            device_batch,
            regularizer,
            self._clip_gradients,
        )

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            with torch.no_grad():
                for target_param, q_network_param in zip(
                    self.target_network.parameters(), self.network.parameters()
                ):
                    target_param.lerp_(q_network_param, self.tau)

        epsilon = linear_schedule(
            self.epsilon_start,
            self.epsilon_end,
            int(self.epsilon_fraction * self.total_timesteps),
            self.global_step,
        )

        if metrics_sink is not None:
            metrics_sink(diagnostics, epsilon, regularizer is not None)
            return {}
        return self.format_update_metrics(diagnostics.tolist(), epsilon, regularizer is not None)

    def format_update_metrics(self, diagnostics, epsilon, regularized):
        """Format ordered CPU diagnostics, retaining the original rolling Q average."""
        loss_value, mean_q_value, penalty_value = diagnostics
        self.q_value_history.append(mean_q_value)
        if len(self.q_value_history) > 1000:
            self.q_value_history.pop(0)
        avg_q = (
            sum(self.q_value_history) / len(self.q_value_history) if self.q_value_history else 0.0
        )

        metrics = {"loss": loss_value, "epsilon": epsilon, "q_value": avg_q}
        if regularized:
            metrics["regularization_loss"] = penalty_value
        return metrics

    def checkpoint_state(self) -> dict:
        """Return all state needed to resume DQN training."""
        return {
            "environment_protocol": self.environment_protocol,
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "global_step": self.global_step,
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore a DQN checkpoint, including optimizer and counters when present."""
        self._compute._update_graph = None
        self._compute._update_key = None
        self.environment_protocol = checkpoint.get("environment_protocol", "gymnasium_wrappers_v1")
        if "network" not in checkpoint:
            self.network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(checkpoint)
            return

        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = checkpoint.get("update_count", 0)
        self.global_step = checkpoint.get("global_step", 0)


class MultiHeadDQNAgent(BaseAgent):
    """DQN agent with shared AtariBackbone and per-task output heads.

    Used for continual learning where different games may have different
    action spaces but share the same convolutional backbone.
    """

    def __init__(
        self,
        state_dim: int,
        lr: float = 1e-4,  # Same as DQNAgent for consistency
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim=1, device=device)

        self.backbone = AtariBackbone(input_channels=state_dim).to(self.device)
        self.target_backbone = AtariBackbone(input_channels=state_dim).to(self.device)
        self.target_backbone.load_state_dict(self.backbone.state_dict())
        self.network = self.backbone

        self.heads = nn.ModuleDict()
        self.target_heads = nn.ModuleDict()

        self.lr = lr
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.task_epsilons: dict[str, float] = {}
        self.update_count = 0
        self.target_update_freq = 1000
        self.optimizer = None
        self.current_task = None
        self.configure_runtime(compile_enabled=False)

    def configure_runtime(
        self, *, compile_enabled: bool = False, capture_updates: bool = True
    ) -> None:
        """Cache separate callables for task heads with different action dimensions."""
        self._compiled = compile_enabled
        self._capture_updates = capture_updates
        self._computations = {}
        self._gradient_groups = (
            tuple(self.backbone.parameters()),
            *(tuple(head.parameters()) for head in self.heads.values()),
        )
        self._clip_gradients = (
            torch.compile(_clip_dqn_gradients, mode="reduce-overhead", fullgraph=True)
            if compile_enabled
            else _clip_dqn_gradients
        )

    def _computation(self):
        task = self.current_task
        if task not in self._computations:
            head, target_head = self._current_heads()
            self._computations[task] = _DQNComputation(
                self.backbone,
                self.target_backbone,
                self.gamma,
                head=head,
                target_head=target_head,
                compiled=self._compiled,
                capture_updates=self._capture_updates,
            )
        return self._computations[task]

    def _rebuild_optimizer(self):
        """Recreate optimizer over backbone and all heads."""
        params = list(self.backbone.parameters())
        for head in self.heads.values():
            params += list(head.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, fused=self.device.type == "cuda")

    def register_task(self, task_id: str, action_dim: int):
        """Create a new output head for a task if it does not exist."""
        if task_id in self.heads:
            return
        self._computations.clear()

        head = nn.Linear(self.backbone.feature_dim, action_dim).to(self.device)
        target_head = nn.Linear(self.backbone.feature_dim, action_dim).to(self.device)
        target_head.load_state_dict(head.state_dict())

        self.heads[task_id] = head
        self.target_heads[task_id] = target_head
        self._gradient_groups = (
            tuple(self.backbone.parameters()),
            *(tuple(task_head.parameters()) for task_head in self.heads.values()),
        )
        self.task_epsilons[task_id] = self.initial_epsilon
        if self.optimizer is None:
            self._rebuild_optimizer()
        else:
            self.optimizer.add_param_group({"params": head.parameters()})

    def set_task(self, task_id: str):
        """Select which task/head to use for subsequent calls."""
        if task_id not in self.heads:
            raise ValueError(f"Task '{task_id}' not registered in MultiHeadDQNAgent.")
        self.current_task = task_id
        self.action_dim = self.heads[task_id].out_features

    def _current_heads(self):
        if self.current_task is None:
            raise RuntimeError("Current task is not set for MultiHeadDQNAgent.")
        return self.heads[self.current_task], self.target_heads[self.current_task]

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection for the current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before select_action().")

        epsilon = self.task_epsilons[self.current_task]
        if not deterministic and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.inference_mode():
            compute = self._computation()
            compute.begin_step()
            state = state.unsqueeze(0).to(self.device, non_blocking=True)
            return compute.greedy(state).item()

    def select_actions(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return epsilon_greedy_batch(
            self._computation(),
            states,
            self.task_epsilons[self.current_task],
            self.action_dim,
            self.device,
            deterministic,
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        regularizer: Callable[[], torch.Tensor] | None = None,
        *,
        metrics_sink: Callable | None = None,
    ) -> dict[str, float]:
        """DQN update using shared backbone and task-specific head."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before update().")
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized; call register_task() first.")

        compute = self._computation()
        device_batch = {
            name: value.to(self.device, non_blocking=True) for name, value in batch.items()
        }
        diagnostics = compute.update(
            self.optimizer,
            self._gradient_groups,
            1.0,
            device_batch,
            regularizer,
            self._clip_gradients,
        )

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_backbone.load_state_dict(self.backbone.state_dict())
            for name, src_head in self.heads.items():
                self.target_heads[name].load_state_dict(src_head.state_dict())

        epsilon = max(
            self.epsilon_min,
            self.task_epsilons[self.current_task] * self.epsilon_decay,
        )
        self.task_epsilons[self.current_task] = epsilon

        if metrics_sink is not None:
            metrics_sink(diagnostics, epsilon, regularizer is not None)
            return {}
        return self.format_update_metrics(diagnostics.tolist(), epsilon, regularizer is not None)

    def format_update_metrics(self, diagnostics, epsilon, regularized):
        """Format diagnostics after the learner's GPU work has completed."""
        loss_value, _, penalty_value = diagnostics
        metrics = {"loss": loss_value, "epsilon": epsilon}
        if regularized:
            metrics["regularization_loss"] = penalty_value
        return metrics

    def checkpoint_state(self) -> dict:
        """Return the shared network, all task heads, and training state."""
        return {
            "environment_protocol": self.environment_protocol,
            "backbone": self.backbone.state_dict(),
            "target_backbone": self.target_backbone.state_dict(),
            "heads": {name: head.state_dict() for name, head in self.heads.items()},
            "target_heads": {name: head.state_dict() for name, head in self.target_heads.items()},
            "task_action_dims": {name: int(head.out_features) for name, head in self.heads.items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "current_task": self.current_task,
            "task_epsilons": self.task_epsilons,
            "update_count": self.update_count,
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore a multi-head DQN checkpoint."""
        self.environment_protocol = checkpoint.get("environment_protocol", "gymnasium_wrappers_v1")
        if "backbone" not in checkpoint:
            # Compatibility with the old, incomplete backbone-only checkpoint.
            self.backbone.load_state_dict(checkpoint)
            self.target_backbone.load_state_dict(checkpoint)
            return

        self.heads = nn.ModuleDict()
        self.target_heads = nn.ModuleDict()
        self.optimizer = None
        self.current_task = None
        self._computations.clear()
        self._gradient_groups = (tuple(self.backbone.parameters()),)

        for task_id, action_dim in checkpoint.get("task_action_dims", {}).items():
            self.register_task(task_id, action_dim)

        self.backbone.load_state_dict(checkpoint["backbone"])
        self.target_backbone.load_state_dict(checkpoint["target_backbone"])
        for task_id, state in checkpoint.get("heads", {}).items():
            self.heads[task_id].load_state_dict(state)
        for task_id, state in checkpoint.get("target_heads", {}).items():
            self.target_heads[task_id].load_state_dict(state)

        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "task_epsilons" in checkpoint:
            self.task_epsilons.update(checkpoint["task_epsilons"])
        elif "epsilon" in checkpoint:
            self.task_epsilons = {task_id: checkpoint["epsilon"] for task_id in self.task_epsilons}
        self.update_count = checkpoint.get("update_count", 0)
        current_task = checkpoint.get("current_task")
        if current_task in self.heads:
            self.set_task(current_task)
