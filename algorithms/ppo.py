"""PPO (Proximal Policy Optimization) implementation."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from .base import BaseAgent, layer_init


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

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value."""
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using policy network."""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            if deterministic:
                hidden = self.network(state / 255.0)
                return self.actor(hidden).argmax(dim=1).item()
            action, _, _, _ = self.get_action_and_value(state)
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
    ) -> dict[str, float]:
        """Update PPO with rollout data."""
        states = rollout_data["states"].to(self.device)
        actions = rollout_data["actions"].to(self.device)
        old_log_probs = rollout_data["log_probs"].to(self.device)
        rewards = rollout_data["rewards"].to(self.device)
        dones = rollout_data["dones"].to(self.device)
        old_values = rollout_data["values"].to(self.device)

        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        b_obs = states
        b_actions = actions.flatten()
        b_log_probs = old_log_probs.flatten()
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()
        b_values = old_values.flatten()

        clipfracs = []
        parameters = [
            *self.network.parameters(),
            *self.actor.parameters(),
            *self.critic.parameters(),
        ]

        for epoch in range(update_epochs):
            b_inds = torch.randperm(len(b_obs), device=self.device)
            for start in range(0, len(b_obs), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_probs, entropy, new_values = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_log_probs = new_log_probs.flatten()
                new_values = new_values.flatten()

                logratio = new_log_probs - b_log_probs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                regularization_loss = regularizer() if regularizer is not None else None
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                if regularization_loss is not None:
                    loss = loss + regularization_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

        metrics = {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": sum(clipfracs) / len(clipfracs),
        }
        if regularizer is not None:
            metrics["regularization_loss"] = regularization_loss.detach().item()
        return metrics


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

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before get_value().")
        hidden = self.backbone(x / 255.0)
        _, critic = self._current_heads()
        return critic(hidden)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before get_action_and_value().")
        hidden = self.backbone(x / 255.0)
        actor, critic = self._current_heads()
        logits = actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), critic(hidden)

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
            action, _, _, _ = self.get_action_and_value(state)
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
    ) -> dict[str, float]:
        """Update PPO with rollout data using shared backbone and task-specific heads."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before update().")
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized; call register_task() first.")

        states = rollout_data["states"].to(self.device)
        actions = rollout_data["actions"].to(self.device)
        old_log_probs = rollout_data["log_probs"].to(self.device)
        rewards = rollout_data["rewards"].to(self.device)
        dones = rollout_data["dones"].to(self.device)
        old_values = rollout_data["values"].to(self.device)

        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        b_obs = states
        b_actions = actions.flatten()
        b_log_probs = old_log_probs.flatten()
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()
        b_values = old_values.flatten()

        clipfracs = []
        all_params = list(self.backbone.parameters())
        for actor in self.actors.values():
            all_params.extend(actor.parameters())
        for critic in self.critics.values():
            all_params.extend(critic.parameters())

        for epoch in range(update_epochs):
            b_inds = torch.randperm(len(b_obs), device=self.device)
            for start in range(0, len(b_obs), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_probs, entropy, new_values = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_log_probs = new_log_probs.flatten()
                new_values = new_values.flatten()

                logratio = new_log_probs - b_log_probs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                regularization_loss = regularizer() if regularizer is not None else None
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                if regularization_loss is not None:
                    loss = loss + regularization_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

        metrics = {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": sum(clipfracs) / len(clipfracs),
        }
        if regularizer is not None:
            metrics["regularization_loss"] = regularization_loss.detach().item()
        return metrics

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
