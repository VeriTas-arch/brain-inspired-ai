"""Shared PPO collection and optimization runtime.

The protocol scripts remain responsible for task order, reporting, and checkpointing. This module
owns shared PPO optimization and the vector collector used by single-task and continual training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.optim as optim

from biai.atari.algorithms.ewc import EWCWrapper
from biai.atari.algorithms.ppo import (
    PolicyEvaluator,
    PPOMinibatchLoss,
    generalized_advantage_estimate,
    ppo_minibatch_loss,
)
from biai.atari.environments import VectorStep

from .buffers import RolloutBuffer


class VectorEnvironment(Protocol):
    """Environment operations required by the PPO collector."""

    num_envs: int

    def reset(self) -> torch.Tensor: ...

    def step_and_reset(self, actions: torch.Tensor) -> VectorStep: ...


class PPOPolicy(Protocol):
    """Agent operations shared by plain and EWC-wrapped PPO agents."""

    device: torch.device
    gamma: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    gae_lambda: float
    max_grad_norm: float
    optimizer: optim.Optimizer

    def sample_action_and_value(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_value(self, state: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CollectedRollout:
    """A time-ordered rollout and the completed episodes observed during collection.

    For T steps and N > 1 environments, data contains states [T, N, C, H, W]
    and actions, log_probs, values, rewards, dones [T, N]. With one environment,
    the N axis is omitted. next_value has shape [N]; transition_count is T * N.
    Keep the time/environment axes until GAE is computed, then flatten for SGD.

    States retain pixel values in [0, 255] on the policy device. Log probabilities
    and values also stay there; actions, rewards, and dones are on CPU. Rewards
    include time-limit bootstrap corrections, while episode_returns are raw
    environment returns. dones marks either termination or truncation.
    The data tensors are buffer views; the next collect call reuses their storage.
    """

    data: dict[str, torch.Tensor]
    next_value: torch.Tensor
    transition_count: int
    episode_returns: tuple[float, ...]


def flatten_rollout_data(
    rollout_data: dict[str, torch.Tensor], *, clone: bool = False
) -> dict[str, torch.Tensor]:
    """Merge [T, N, ...] into [T * N, ...], preserving time-major order.

    Single-environment data already has a single leading axis. Use clone=True
    to retain a batch after the collector reuses its buffer. This flattened
    representation is for boundary sampling; PPO computes GAE before flattening.
    """
    states = rollout_data["states"]
    has_environment_axis = states.ndim == 5
    flattened = {
        name: value.flatten(0, 1) if has_environment_axis else value
        for name, value in rollout_data.items()
    }
    if clone:
        return {name: value.detach().clone() for name, value in flattened.items()}
    return flattened


def configure_ppo_runtime(environment_backend: str) -> None:
    """Avoid CPU thread-pool overhead with CUDA policies or environment workers."""
    import cv2

    cv2.setNumThreads(1)
    if environment_backend in {"async", "ale"} or torch.cuda.is_available():
        torch.set_num_threads(1)


def _clip_ppo_gradients(parameters, max_norm):
    nn.utils.clip_grad_norm_(parameters, max_norm)


def indexed_ppo_loss(
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    old_values,
    indices,
    regularizer,
    *,
    policy_evaluator,
    clip_coef,
    ent_coef,
    vf_coef,
):
    """Gather a minibatch and compute its objective and owned, detached diagnostics."""
    loss, *metrics = ppo_minibatch_loss(
        states[indices],
        actions[indices],
        old_log_probs[indices],
        advantages[indices],
        returns[indices],
        old_values[indices],
        policy_evaluator=policy_evaluator,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
    )
    penalty = loss.new_zeros(()) if regularizer is None else regularizer()
    return loss + penalty, torch.stack([value.detach() for value in (*metrics, penalty)])


def optimize_ppo(
    rollout_data: dict[str, torch.Tensor],
    next_value: torch.Tensor,
    *,
    device: torch.device,
    optimizer: optim.Optimizer,
    parameters: list[nn.Parameter],
    policy_evaluator: PolicyEvaluator,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    update_epochs: int,
    minibatch_size: int,
    regularizer: Callable[[], torch.Tensor] | None = None,
    minibatch_loss: PPOMinibatchLoss | None = None,
    clip_gradients: Callable | None = None,
    captured_update: Callable | None = None,
) -> dict[str, float]:
    """Run the PPO learner shared by single-head and multi-head agents."""
    if update_epochs <= 0:
        raise ValueError("update_epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if len(rollout_data["states"]) == 0:
        raise ValueError("PPO update requires a non-empty rollout")

    states = rollout_data["states"].to(device, non_blocking=True)
    actions = rollout_data["actions"].to(device, non_blocking=True)
    old_log_probs = rollout_data["log_probs"].to(device, non_blocking=True)
    # The collector already keeps rewards on CPU. Run the short time-axis recurrence
    # there rather than launching many tiny CUDA kernels; GPU-native rollouts stay on GPU.
    rewards = rollout_data["rewards"]
    dones = rollout_data["dones"].to(rewards.device)
    old_values = rollout_data["values"].to(rewards.device)

    with torch.no_grad():
        advantages, returns = generalized_advantage_estimate(
            rewards,
            old_values,
            dones,
            next_value.to(rewards.device),
            gamma,
            gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    advantages = advantages.to(device)
    returns = returns.to(device)
    old_values = old_values.to(device)

    batch_states = states.flatten(0, 1) if states.ndim == 5 else states
    batch_actions = actions.flatten()
    batch_log_probs = old_log_probs.flatten()
    batch_advantages = advantages.flatten()
    batch_returns = returns.flatten()
    batch_values = old_values.flatten()
    loss_arguments = (
        batch_states,
        batch_actions,
        batch_log_probs,
        batch_advantages,
        batch_returns,
        batch_values,
    )
    clip_gradients = clip_gradients or nn.utils.clip_grad_norm_

    metric_sums = torch.zeros(6, device=device)
    update_count = 0
    used_regularizer = regularizer is not None

    for _ in range(update_epochs):
        batch_indices = torch.randperm(len(batch_states), device=device)
        for start in range(0, len(batch_states), minibatch_size):
            minibatch_indices = batch_indices[start : start + minibatch_size]
            if captured_update is not None and len(minibatch_indices) == minibatch_size:
                diagnostics = captured_update(
                    (*loss_arguments, minibatch_indices), regularizer, update_count == 0
                )
                metric_sums += diagnostics
                update_count += 1
                continue
            if minibatch_loss is None or len(minibatch_indices) < minibatch_size:
                loss, diagnostics = indexed_ppo_loss(
                    *loss_arguments,
                    minibatch_indices,
                    regularizer,
                    policy_evaluator=policy_evaluator,
                    clip_coef=clip_coef,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                )
            else:
                loss, diagnostics = minibatch_loss(*loss_arguments, minibatch_indices, regularizer)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(parameters, max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metric_sums += diagnostics
                update_count += 1

    averages = (metric_sums / update_count).tolist()
    metrics = dict(
        zip(
            ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"),
            averages[:5],
            strict=True,
        )
    )
    if used_regularizer:
        metrics["regularization_loss"] = averages[5]
    return metrics


class PPOLearner:
    """One PPO optimization path with optional compiled sampling and minibatch loss."""

    def __init__(
        self,
        agent: PPOPolicy,
        *,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        compile_policy: bool = False,
        regularizer: Callable[[], torch.Tensor] | None = None,
        capture_updates: bool = True,
    ) -> None:
        if update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")

        self.agent = agent
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.compiled = compile_policy
        self.capture_updates = compile_policy and capture_updates and agent.device.type == "cuda"
        self._update_graphs = {}
        self._update_generation = None
        self.regularizer = regularizer

        if isinstance(agent, EWCWrapper):
            if regularizer is not None:
                raise ValueError("Use only one PPO regularizer")
            agent.configure_regularizer()

        policy_agent = getattr(agent, "agent", agent)
        sampler = policy_agent.sample_action_and_value

        def minibatch_loss(
            states: torch.Tensor,
            actions: torch.Tensor,
            old_log_probs: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_values: torch.Tensor,
            indices: torch.Tensor,
            regularizer: Callable | None,
        ):
            return indexed_ppo_loss(
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
                old_values,
                indices,
                regularizer,
                policy_evaluator=policy_agent.get_action_and_value,
                clip_coef=policy_agent.clip_coef,
                ent_coef=policy_agent.ent_coef,
                vf_coef=policy_agent.vf_coef,
            )

        if compile_policy:
            sampler = torch.compile(sampler, mode="reduce-overhead", fullgraph=True)
            minibatch_loss = torch.compile(minibatch_loss, mode="reduce-overhead", fullgraph=True)
        self._sample_action_and_value = sampler
        self._loss = minibatch_loss
        self._clip_gradients = (
            # Parameter shapes are fixed within a task. Symbolic generalization of
            # parameter.grad across different task heads loses shape sources in PyTorch 2.14.
            torch.compile(
                _clip_ppo_gradients, mode="reduce-overhead", fullgraph=True, dynamic=False
            )
            if compile_policy
            else _clip_ppo_gradients
        )
        self._policy_evaluator = policy_agent.get_action_and_value

    @property
    def device(self) -> torch.device:
        """Return the policy device."""
        return self.agent.device

    @property
    def gamma(self) -> float:
        """Return the policy discount factor."""
        return self.agent.gamma

    def sample_action_and_value(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions through the eager or compiled rollout policy."""
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        return self._sample_action_and_value(state)

    def _minibatch_loss(self, *args):
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        return self._loss(*args)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate bootstrap values through the underlying policy."""
        return self.agent.get_value(state)

    def _captured_update(self, inputs, regularizer, refresh):
        from .cuda_update import CudaUpdate

        agent = getattr(self.agent, "agent", self.agent)
        if self._update_generation != agent._update_generation:
            self._update_graphs.clear()
            self._update_generation = agent._update_generation
        key = (
            agent._update_generation,
            getattr(agent, "current_task", None),
            regularizer,
            getattr(agent.optimizer, "_subspace_projection", None),
            tuple((value.shape, value.dtype) for value in inputs),
        )
        if key not in self._update_graphs:
            parameters = [p for group in agent.optimizer.param_groups for p in group["params"]]

            def loss(*values):
                return indexed_ppo_loss(
                    *values,
                    regularizer,
                    policy_evaluator=agent.get_action_and_value,
                    clip_coef=agent.clip_coef,
                    ent_coef=agent.ent_coef,
                    vf_coef=agent.vf_coef,
                )

            self._update_graphs[key] = CudaUpdate(
                agent.optimizer,
                loss,
                lambda: _clip_ppo_gradients(parameters, agent.max_grad_norm),
                inputs,
            )
        return self._update_graphs[key](*inputs, refresh_prefix=refresh)

    def update(self, rollout: CollectedRollout) -> dict[str, float]:
        """Optimize one collected rollout."""
        policy = getattr(self.agent, "agent", self.agent)
        regularizer = (
            self.agent._runtime_regularizer
            if isinstance(self.agent, EWCWrapper)
            else self.regularizer
        )
        metrics = optimize_ppo(
            rollout.data,
            rollout.next_value,
            device=policy.device,
            optimizer=policy.optimizer,
            parameters=[p for group in policy.optimizer.param_groups for p in group["params"]],
            policy_evaluator=self._policy_evaluator,
            gamma=policy.gamma,
            gae_lambda=policy.gae_lambda,
            clip_coef=policy.clip_coef,
            ent_coef=policy.ent_coef,
            vf_coef=policy.vf_coef,
            max_grad_norm=policy.max_grad_norm,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            regularizer=regularizer,
            minibatch_loss=self._minibatch_loss,
            clip_gradients=self._clip_gradients,
            captured_update=self._captured_update if self.capture_updates else None,
        )
        if isinstance(self.agent, EWCWrapper) and "regularization_loss" in metrics:
            metrics["ewc_loss"] = metrics.pop("regularization_loss")
        return metrics


class PPOCollector:
    """Collect independent vector trajectories for a PPO learner."""

    def __init__(
        self,
        environment: VectorEnvironment,
        learner: PPOLearner,
        *,
        rollout_length: int = 128,
        frame_callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> None:
        if rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        self.environment = environment
        self.learner = learner
        self.num_envs = environment.num_envs
        self.buffer = RolloutBuffer(
            capacity=rollout_length, num_envs=self.num_envs, policy_device=learner.device
        )
        self.state = environment.reset()
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float64)
        self.transition_count = 0
        self.episode_count = 0
        self.frame_callback = frame_callback

    def collect(self, transition_budget: int) -> CollectedRollout:
        """Collect up to one rollout without exceeding an exact transition budget."""
        if transition_budget <= 0:
            raise ValueError("transition_budget must be positive")
        if transition_budget % self.num_envs != 0:
            raise ValueError("transition_budget must be divisible by num_envs")

        vector_steps = min(self.buffer.capacity, transition_budget // self.num_envs)
        self.buffer.reset()
        completed_returns: list[float] = []

        for _ in range(vector_steps):
            with torch.inference_mode():
                device_state = self.state.to(self.learner.device, non_blocking=True)
                actions, log_probs, values = self.learner.sample_action_and_value(device_state)
            cpu_actions = actions.cpu()

            if self.frame_callback is not None:
                self.frame_callback(self.transition_count, self.state[0, 0])

            if self.num_envs == 1:
                self.buffer.start_step(
                    device_state[0],
                    cpu_actions.item(),
                    log_probs.flatten()[0],
                    values.flatten()[0],
                )
            else:
                self.buffer.start_step(
                    device_state,
                    cpu_actions,
                    log_probs,
                    values.flatten(),
                )

            transition = self.environment.step_and_reset(cpu_actions)
            done = transition.terminated | transition.truncated
            training_rewards = transition.rewards.clone()
            bootstrap_mask = transition.truncated & ~transition.terminated
            if bootstrap_mask.any():
                with torch.inference_mode():
                    truncated_values = self.learner.get_value(
                        transition.transition_observations[bootstrap_mask].to(self.learner.device)
                    ).flatten()
                training_rewards[bootstrap_mask] += self.learner.gamma * truncated_values.cpu()

            if self.num_envs == 1:
                self.buffer.finish_step(training_rewards.item(), done.item())
            else:
                self.buffer.finish_step(training_rewards, done)

            self.episode_returns += transition.rewards.to(torch.float64)
            for index in done.nonzero().flatten().tolist():
                completed_returns.append(self.episode_returns[index].item())
                self.episode_returns[index] = 0.0
                self.episode_count += 1

            self.state = transition.observations
            self.transition_count += self.num_envs

        with torch.inference_mode():
            next_value = self.learner.get_value(self.state.to(self.learner.device)).flatten()

        return CollectedRollout(
            data=self.buffer.get_batch(),
            next_value=next_value,
            transition_count=self.buffer.transition_count,
            episode_returns=tuple(completed_returns),
        )
