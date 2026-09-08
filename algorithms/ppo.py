"""PPO (Proximal Policy Optimization) implementation."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from .base import BaseAgent, layer_init

PolicyEvaluator = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
PPOMinibatchLoss = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]


def bootstrap_truncated_reward(
    reward: float,
    next_value: float,
    *,
    terminated: bool,
    truncated: bool,
    gamma: float,
) -> float:
    """Bootstrap a time-limit transition while still ending its GAE segment."""
    if truncated and not terminated:
        return reward + gamma * next_value
    return reward


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE when ``dones[t]`` describes the transition at index ``t``."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)

    for t in reversed(range(len(rewards))):
        next_values = next_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[t] = last_advantage

    return advantages, advantages + values


def ppo_minibatch_loss(
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    *,
    policy_evaluator: PolicyEvaluator,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute one PPO minibatch objective and its diagnostics."""
    _, new_log_probs, entropy, new_values = policy_evaluator(states, actions)
    new_log_probs = new_log_probs.flatten()
    new_values = new_values.flatten()

    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    policy_loss = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
    ).mean()

    value_loss_unclipped = (new_values - returns).square()
    clipped_values = old_values + torch.clamp(
        new_values - old_values,
        -clip_coef,
        clip_coef,
    )
    value_loss_clipped = (clipped_values - returns).square()
    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

    entropy_loss = entropy.mean()
    loss = policy_loss - ent_coef * entropy_loss + vf_coef * value_loss
    approximate_kl = ((ratio - 1) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return loss, policy_loss, value_loss, entropy_loss, approximate_kl, clip_fraction


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
) -> dict[str, float]:
    """Run the PPO learner shared by single-head and multi-head agents."""
    if update_epochs <= 0:
        raise ValueError("update_epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if len(rollout_data["states"]) == 0:
        raise ValueError("PPO update requires a non-empty rollout")

    states = rollout_data["states"].to(device)
    actions = rollout_data["actions"].to(device)
    old_log_probs = rollout_data["log_probs"].to(device)
    rewards = rollout_data["rewards"].to(device)
    dones = rollout_data["dones"].to(device)
    old_values = rollout_data["values"].to(device)

    with torch.no_grad():
        advantages, returns = generalized_advantage_estimate(
            rewards,
            old_values,
            dones,
            next_value.to(device),
            gamma,
            gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch_states = states.flatten(0, 1) if states.ndim == 5 else states
    batch_actions = actions.flatten()
    batch_log_probs = old_log_probs.flatten()
    batch_advantages = advantages.flatten()
    batch_returns = returns.flatten()
    batch_values = old_values.flatten()

    metric_sums = torch.zeros(6, device=device)
    update_count = 0
    used_regularizer = regularizer is not None

    for _ in range(update_epochs):
        batch_indices = torch.randperm(len(batch_states), device=device)
        for start in range(0, len(batch_states), minibatch_size):
            minibatch_indices = batch_indices[start : start + minibatch_size]
            loss_arguments = (
                batch_states[minibatch_indices],
                batch_actions[minibatch_indices],
                batch_log_probs[minibatch_indices],
                batch_advantages[minibatch_indices],
                batch_returns[minibatch_indices],
                batch_values[minibatch_indices],
            )
            if minibatch_loss is None or len(minibatch_indices) < minibatch_size:
                (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    approximate_kl,
                    clip_fraction,
                ) = ppo_minibatch_loss(
                    *loss_arguments,
                    policy_evaluator=policy_evaluator,
                    clip_coef=clip_coef,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                )
            else:
                (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    approximate_kl,
                    clip_fraction,
                ) = minibatch_loss(*loss_arguments)

            regularization_loss = regularizer() if regularizer is not None else None
            if regularization_loss is not None:
                loss = loss + regularization_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                regularization_metric = (
                    regularization_loss
                    if regularization_loss is not None
                    else torch.zeros((), device=device)
                )
                metric_sums += torch.stack(
                    (
                        policy_loss,
                        value_loss,
                        entropy_loss,
                        approximate_kl,
                        clip_fraction,
                        regularization_metric,
                    )
                )
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


class PPOAgent(BaseAgent):
    """PPO Agent for Atari games."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, device)

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(state_dim, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        ).to(self.device)

        self.actor = layer_init(nn.Linear(512, action_dim), std=0.01).to(self.device)
        self.critic = layer_init(nn.Linear(512, 1), std=1).to(self.device)

        self.optimizer = optim.Adam(
            list(self.network.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def checkpoint_state(self) -> dict:
        """Return all state needed to resume PPO training."""
        return {
            "network": self.network.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore a PPO checkpoint."""
        self.network.load_state_dict(checkpoint["network"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.critic(self.network(x / 255.0))

    def _distribution_and_value(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        hidden = self.network(x / 255.0)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden)

    def sample_action_and_value(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample rollout data without computing the unused policy entropy."""
        distribution, value = self._distribution_and_value(x)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value."""
        probs, value = self._distribution_and_value(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using policy network."""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            if deterministic:
                hidden = self.network(state / 255.0)
                return self.actor(hidden).argmax(dim=1).item()
            action, _, _ = self.sample_action_and_value(state)
            return action.item()

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ):
        """Compute GAE advantages and returns."""
        return generalized_advantage_estimate(
            rewards,
            values,
            dones,
            next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def update(
        self,
        rollout_data: dict[str, torch.Tensor],
        next_value: torch.Tensor,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        regularizer: Callable[[], torch.Tensor] | None = None,
        policy_evaluator: PolicyEvaluator | None = None,
        minibatch_loss: PPOMinibatchLoss | None = None,
    ) -> dict[str, float]:
        """Update PPO with rollout data."""
        parameters = [
            *self.network.parameters(),
            *self.actor.parameters(),
            *self.critic.parameters(),
        ]
        return optimize_ppo(
            rollout_data,
            next_value,
            device=self.device,
            optimizer=self.optimizer,
            parameters=parameters,
            policy_evaluator=policy_evaluator or self.get_action_and_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            regularizer=regularizer,
            minibatch_loss=minibatch_loss,
        )


class MultiHeadPPOAgent(BaseAgent):
    """PPO agent with shared AtariBackbone and per-task actor/critic heads.

    Used for continual learning where different games may have different
    action spaces but share the same convolutional backbone.
    """

    def __init__(
        self,
        state_dim: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim=1, device=device)

        # Shared backbone (same as PPOAgent's network)
        self.backbone = nn.Sequential(
            layer_init(nn.Conv2d(state_dim, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        ).to(self.device)
        self.network = self.backbone

        # Per-task actor and critic heads
        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = None
        self.current_task = None

    def _rebuild_optimizer(self):
        """Recreate optimizer over backbone and all heads."""
        params = list(self.backbone.parameters())
        for actor in self.actors.values():
            params += list(actor.parameters())
        for critic in self.critics.values():
            params += list(critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, eps=1e-5)

    def register_task(self, task_id: str, action_dim: int):
        """Create new actor and critic heads for a task if they don't exist."""
        if task_id in self.actors:
            return

        actor = layer_init(nn.Linear(512, action_dim), std=0.01).to(self.device)
        critic = layer_init(nn.Linear(512, 1), std=1).to(self.device)

        self.actors[task_id] = actor
        self.critics[task_id] = critic
        if self.optimizer is None:
            self._rebuild_optimizer()
        else:
            self.optimizer.add_param_group({"params": [*actor.parameters(), *critic.parameters()]})

    def set_task(self, task_id: str):
        """Select which task/heads to use for subsequent calls."""
        if task_id not in self.actors:
            raise ValueError(f"Task '{task_id}' not registered in MultiHeadPPOAgent.")
        self.current_task = task_id
        self.action_dim = self.actors[task_id].out_features

    def _current_heads(self):
        """Get current task's actor and critic heads."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set for MultiHeadPPOAgent.")
        return self.actors[self.current_task], self.critics[self.current_task]

    def policy_logits(self, states: torch.Tensor, task_id: str) -> torch.Tensor:
        """Evaluate an actor without changing the active PPO task or touching its critic."""
        return self.actors[task_id](self.backbone(states / 255.0))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before get_value().")
        hidden = self.backbone(x / 255.0)
        _, critic = self._current_heads()
        return critic(hidden)

    def _distribution_and_value(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if self.current_task is None:
            raise RuntimeError("Current task is not set before policy evaluation.")
        hidden = self.backbone(x / 255.0)
        actor, critic = self._current_heads()
        return Categorical(logits=actor(hidden)), critic(hidden)

    def sample_action_and_value(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample rollout data without computing the unused policy entropy."""
        distribution, value = self._distribution_and_value(x)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value for current task."""
        probs, value = self._distribution_and_value(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using policy network for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before select_action().")
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            if deterministic:
                actor, _ = self._current_heads()
                hidden = self.backbone(state / 255.0)
                return actor(hidden).argmax(dim=1).item()
            action, _, _ = self.sample_action_and_value(state)
            return action.item()

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ):
        """Compute GAE advantages and returns."""
        return generalized_advantage_estimate(
            rewards,
            values,
            dones,
            next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def update(
        self,
        rollout_data: dict[str, torch.Tensor],
        next_value: torch.Tensor,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        regularizer: Callable[[], torch.Tensor] | None = None,
        policy_evaluator: PolicyEvaluator | None = None,
        minibatch_loss: PPOMinibatchLoss | None = None,
    ) -> dict[str, float]:
        """Update PPO with rollout data using shared backbone and task-specific heads."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before update().")
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized; call register_task() first.")
        all_params = list(self.backbone.parameters())
        for actor in self.actors.values():
            all_params.extend(actor.parameters())
        for critic in self.critics.values():
            all_params.extend(critic.parameters())
        return optimize_ppo(
            rollout_data,
            next_value,
            device=self.device,
            optimizer=self.optimizer,
            parameters=all_params,
            policy_evaluator=policy_evaluator or self.get_action_and_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            regularizer=regularizer,
            minibatch_loss=minibatch_loss,
        )

    def checkpoint_state(self) -> dict:
        """Return the shared network, task heads, and optimizer state."""
        return {
            "backbone": self.backbone.state_dict(),
            "actors": {task_id: actor.state_dict() for task_id, actor in self.actors.items()},
            "critics": {task_id: critic.state_dict() for task_id, critic in self.critics.items()},
            "task_action_dims": {
                task_id: actor.out_features for task_id, actor in self.actors.items()
            },
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "current_task": self.current_task,
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore a multi-head PPO checkpoint."""
        if "backbone" not in checkpoint:
            self.backbone.load_state_dict(checkpoint)
            return

        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        self.optimizer = None
        self.current_task = None

        for task_id, action_dim in checkpoint.get("task_action_dims", {}).items():
            self.register_task(task_id, action_dim)
        self.backbone.load_state_dict(checkpoint["backbone"])

        for task_id, state_dict in checkpoint["actors"].items():
            if task_id not in self.actors:
                if "task_action_dims" in checkpoint and task_id in checkpoint["task_action_dims"]:
                    action_dim = checkpoint["task_action_dims"][task_id]
                    self.register_task(task_id, action_dim)
                else:
                    continue
            self.actors[task_id].load_state_dict(state_dict)

        for task_id, state_dict in checkpoint["critics"].items():
            if task_id not in self.critics:
                if "task_action_dims" in checkpoint and task_id in checkpoint["task_action_dims"]:
                    action_dim = checkpoint["task_action_dims"][task_id]
                    if task_id not in self.actors:
                        self.register_task(task_id, action_dim)
                else:
                    continue
            self.critics[task_id].load_state_dict(state_dict)

        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        current_task = checkpoint.get("current_task")
        if current_task in self.actors:
            self.set_task(current_task)
