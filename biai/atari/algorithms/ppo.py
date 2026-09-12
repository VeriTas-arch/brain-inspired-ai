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
PPOMinibatchLoss = Callable[..., tuple[torch.Tensor, torch.Tensor]]


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


def greedy_policy_actions(states, backbone, actor):
    return actor(backbone(states / 255.0)).argmax(dim=1)


def sample_policy_actions(states, backbone, actor):
    """Sample boundary actions without computing values or log probabilities."""
    return Categorical(logits=actor(backbone(states / 255.0)), validate_args=False).sample()


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
            fused=self.device.type == "cuda",
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.configure_runtime()

    def configure_runtime(self, *, compile_enabled: bool = False) -> None:
        """Compile deterministic batched inference used by evaluation."""
        self._update_generation = getattr(self, "_update_generation", 0) + 1
        self.compiled_inference = compile_enabled
        self._greedy_actions = (
            torch.compile(greedy_policy_actions, mode="reduce-overhead", fullgraph=True)
            if compile_enabled
            else greedy_policy_actions
        )

    @torch.no_grad()
    def select_actions(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if not deterministic:
            return sample_policy_actions(states.to(self.device), self.network, self.actor)
        if self.compiled_inference:
            torch.compiler.cudagraph_mark_step_begin()
        return self._greedy_actions(
            states.to(self.device, non_blocking=True), self.network, self.actor
        )

    def checkpoint_state(self) -> dict:
        """Return all state needed to resume PPO training."""
        return {
            "environment_protocol": self.environment_protocol,
            "network": self.network.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore a PPO checkpoint."""
        self._update_generation += 1
        self.environment_protocol = checkpoint.get("environment_protocol", "gymnasium_wrappers_v1")
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
        # These logits and sampled actions are internal; validation synchronizes CUDA.
        return Categorical(logits=self.actor(hidden), validate_args=False), self.critic(hidden)

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
        if deterministic:
            return self.select_actions(state.unsqueeze(0)).item()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action, _, _ = self.sample_action_and_value(state)
            return action.item()


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
        self.configure_runtime()

    def configure_runtime(self, *, compile_enabled: bool = False) -> None:
        """Compile deterministic inference while keeping task selection outside the graph."""
        self._update_generation = getattr(self, "_update_generation", 0) + 1
        self.compiled_inference = compile_enabled
        self._greedy_actions = (
            torch.compile(greedy_policy_actions, mode="reduce-overhead", fullgraph=True)
            if compile_enabled
            else greedy_policy_actions
        )

    @torch.no_grad()
    def select_actions(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        actor, _ = self._current_heads()
        if not deterministic:
            return sample_policy_actions(states.to(self.device), self.backbone, actor)
        if self.compiled_inference:
            torch.compiler.cudagraph_mark_step_begin()
        return self._greedy_actions(states.to(self.device, non_blocking=True), self.backbone, actor)

    def _rebuild_optimizer(self):
        """Recreate optimizer over backbone and all heads."""
        params = list(self.backbone.parameters())
        for actor in self.actors.values():
            params += list(actor.parameters())
        for critic in self.critics.values():
            params += list(critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, eps=1e-5, fused=self.device.type == "cuda")

    def register_task(self, task_id: str, action_dim: int):
        """Create new actor and critic heads for a task if they don't exist."""
        if task_id in self.actors:
            return
        self._update_generation += 1

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

    def _distribution_and_value(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if self.current_task is None:
            raise RuntimeError("Current task is not set before policy evaluation.")
        hidden = self.backbone(x / 255.0)
        actor, critic = self._current_heads()
        # These logits and sampled actions are internal; validation synchronizes CUDA.
        return Categorical(logits=actor(hidden), validate_args=False), critic(hidden)

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
        if deterministic:
            return self.select_actions(state.unsqueeze(0)).item()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action, _, _ = self.sample_action_and_value(state)
            return action.item()

    def checkpoint_state(self) -> dict:
        """Return the shared network, task heads, and optimizer state."""
        return {
            "environment_protocol": self.environment_protocol,
            "backbone": self.backbone.state_dict(),
            "actors": {task_id: actor.state_dict() for task_id, actor in self.actors.items()},
            "critics": {task_id: critic.state_dict() for task_id, critic in self.critics.items()},
            "task_action_dims": {
                task_id: int(actor.out_features) for task_id, actor in self.actors.items()
            },
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "current_task": self.current_task,
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore a multi-head PPO checkpoint."""
        tasks = checkpoint.get("task_action_dims", {})
        if (
            not tasks
            or "backbone" not in checkpoint
            or any(set(checkpoint.get(key, {})) != set(tasks) for key in ("actors", "critics"))
        ):
            raise ValueError(
                "Incomplete multi-head PPO checkpoint: backbone and all task heads are required"
            )
        self._update_generation += 1
        self.environment_protocol = checkpoint.get("environment_protocol", "gymnasium_wrappers_v1")

        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        self.optimizer = None
        self.current_task = None

        for task_id, action_dim in tasks.items():
            self.register_task(task_id, action_dim)
        self.backbone.load_state_dict(checkpoint["backbone"])

        for task_id, state_dict in checkpoint["actors"].items():
            self.actors[task_id].load_state_dict(state_dict)

        for task_id, state_dict in checkpoint["critics"].items():
            self.critics[task_id].load_state_dict(state_dict)

        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        current_task = checkpoint.get("current_task")
        if current_task in self.actors:
            self.set_task(current_task)
