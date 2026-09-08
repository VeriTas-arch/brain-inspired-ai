"""Shared PPO collection and optimization runtime.

The protocol scripts remain responsible for task order, reporting, and checkpointing. This module
owns only the mechanics that must be identical between single-task and continual PPO training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from algorithms.ppo import ppo_minibatch_loss
from environments import VectorStep

from .buffers import RolloutBuffer


class VectorEnvironment(Protocol):
    """Environment operations required by the PPO collector."""

    num_envs: int

    def reset(self) -> torch.Tensor: ...

    def step_and_reset(self, actions: torch.Tensor) -> VectorStep: ...


class PPOAgent(Protocol):
    """Agent operations shared by plain and EWC-wrapped PPO agents."""

    device: torch.device
    gamma: float
    clip_coef: float
    ent_coef: float
    vf_coef: float

    def sample_action_and_value(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_value(self, state: torch.Tensor) -> torch.Tensor: ...

    def update(
        self,
        rollout_data: dict[str, torch.Tensor],
        next_value: torch.Tensor,
        update_epochs: int,
        minibatch_size: int,
        **kwargs,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class CollectedRollout:
    """A rollout batch and the events observed while collecting it."""

    data: dict[str, torch.Tensor]
    next_value: torch.Tensor
    transition_count: int
    episode_returns: tuple[float, ...]


def flatten_rollout_data(
    rollout_data: dict[str, torch.Tensor], *, clone: bool = False
) -> dict[str, torch.Tensor]:
    """Flatten time and environment axes after GAE-compatible collection."""
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
    """Avoid CPU thread-pool contention with asynchronous environment workers."""
    if environment_backend == "async":
        torch.set_num_threads(1)


class PPOLearner:
    """One PPO optimization path with optional compiled sampling and minibatch loss."""

    def __init__(
        self,
        agent: PPOAgent,
        *,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        compile_policy: bool = False,
        regularizer: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        if update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")

        self.agent = agent
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.compiled = compile_policy
        self.regularizer = regularizer

        configure_regularizer = getattr(agent, "configure_regularizer", None)
        if configure_regularizer is not None and regularizer is not None:
            raise ValueError("Use only one PPO regularizer")
        if configure_regularizer is not None:
            configure_regularizer(compile_regularizer=compile_policy)

        policy_agent = getattr(agent, "agent", agent)
        sampler = policy_agent.sample_action_and_value

        def minibatch_loss(
            states: torch.Tensor,
            actions: torch.Tensor,
            old_log_probs: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_values: torch.Tensor,
        ):
            return ppo_minibatch_loss(
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
                old_values,
                policy_evaluator=policy_agent.get_action_and_value,
                clip_coef=policy_agent.clip_coef,
                ent_coef=policy_agent.ent_coef,
                vf_coef=policy_agent.vf_coef,
            )

        if compile_policy:
            sampler = torch.compile(sampler, mode="reduce-overhead")
            minibatch_loss = torch.compile(minibatch_loss, mode="reduce-overhead")
        self._sample_action_and_value = sampler
        self._minibatch_loss = minibatch_loss
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
        return self._sample_action_and_value(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate bootstrap values through the underlying policy."""
        return self.agent.get_value(state)

    def update(self, rollout: CollectedRollout) -> dict[str, float]:
        """Optimize one collected rollout."""
        extra = {} if self.regularizer is None else {"regularizer": self.regularizer}
        return self.agent.update(
            rollout.data,
            rollout.next_value,
            self.update_epochs,
            self.minibatch_size,
            policy_evaluator=self._policy_evaluator,
            minibatch_loss=self._minibatch_loss,
            **extra,
        )


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
        self.buffer = RolloutBuffer(capacity=rollout_length, num_envs=self.num_envs)
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
                device_state = self.state.to(self.learner.device)
                actions, log_probs, values = self.learner.sample_action_and_value(device_state)
            cpu_actions = actions.cpu()

            if self.frame_callback is not None:
                self.frame_callback(self.transition_count, self.state[0, 0])

            if self.num_envs == 1:
                self.buffer.start_step(
                    self.state[0],
                    cpu_actions.item(),
                    log_probs.item(),
                    values.item(),
                )
            else:
                self.buffer.start_step(
                    self.state,
                    cpu_actions,
                    log_probs.cpu(),
                    values.flatten().cpu(),
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
