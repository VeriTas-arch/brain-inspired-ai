"""Elastic Weight Consolidation for the teaching implementations."""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import BaseAgent, safe_torch_load


def summarize_parameter_importance(
    importance: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    """Summarize diagonal parameter-importance coverage by top-level module."""
    modules: dict[str, dict[str, float | int]] = {}
    for name, values in importance.items():
        module = name.split(".", 1)[0]
        summary = modules.setdefault(
            module,
            {
                "parameter_tensors": 0,
                "elements": 0,
                "nonzero_elements": 0,
                "importance_sum": 0.0,
                "importance_max": 0.0,
            },
        )
        summary["parameter_tensors"] += 1
        summary["elements"] += values.numel()
        summary["nonzero_elements"] += torch.count_nonzero(values).item()
        summary["importance_sum"] += values.sum().item()
        summary["importance_max"] = max(float(summary["importance_max"]), values.max().item())

    for summary in modules.values():
        elements = int(summary["elements"])
        summary["nonzero_fraction"] = (
            int(summary["nonzero_elements"]) / elements if elements else 0.0
        )
    return modules


class EWCWrapper:
    """Add an EWC regularizer without duplicating an agent's update loop."""

    def __init__(self, agent: BaseAgent, ewc_lambda: float = 0.4):
        self.agent = agent
        self.ewc_lambda = ewc_lambda
        self.task_weights: dict[int, dict[str, torch.Tensor]] = {}
        self.task_fisher: dict[int, dict[str, torch.Tensor]] = {}
        self.task_fisher_summary: dict[int, dict] = {}
        self.current_task_id = 0

    def _collect_regularized_params(self) -> dict[str, torch.Tensor]:
        """Collect unique online parameters relevant to the current task."""
        params: dict[str, torch.Tensor] = {}
        seen: set[int] = set()

        def add_module(prefix: str, module) -> None:
            if module is None:
                return
            for name, parameter in module.named_parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    params[f"{prefix}.{name}"] = parameter
                    seen.add(id(parameter))

        backbone = getattr(self.agent, "backbone", None)
        if backbone is not None:
            add_module("backbone", backbone)
        else:
            add_module("network", getattr(self.agent, "network", None))

        current_task = getattr(self.agent, "current_task", None)
        heads = getattr(self.agent, "heads", None)
        if heads is not None and current_task in heads:
            add_module(f"heads.{current_task}", heads[current_task])

        actors = getattr(self.agent, "actors", None)
        if actors is not None and current_task in actors:
            add_module(f"actors.{current_task}", actors[current_task])
        else:
            add_module("actor", getattr(self.agent, "actor", None))

        return params

    def _dqn_sample_loss(self, batch: dict[str, torch.Tensor], index: int) -> torch.Tensor:
        states = batch["states"][index : index + 1].to(self.agent.device)
        actions = batch["actions"][index : index + 1].to(self.agent.device).long()
        rewards = batch["rewards"][index : index + 1].to(self.agent.device).flatten()
        next_states = batch["next_states"][index : index + 1].to(self.agent.device)
        dones = batch["dones"][index : index + 1].to(self.agent.device).flatten()

        if hasattr(self.agent, "heads"):
            head, target_head = self.agent._current_heads()
            q_values = head(self.agent.backbone(states)).gather(1, actions.view(-1, 1)).flatten()
            with torch.no_grad():
                next_features = self.agent.target_backbone(next_states)
                next_q_values = target_head(next_features).max(dim=1).values
        else:
            q_values = self.agent.network(states).gather(1, actions.view(-1, 1)).flatten()
            with torch.no_grad():
                next_q_values = self.agent.target_network(next_states).max(dim=1).values

        targets = rewards + self.agent.gamma * next_q_values * (1.0 - dones)
        return F.mse_loss(q_values, targets)

    def _ppo_sample_loss(self, batch: dict[str, torch.Tensor], index: int) -> torch.Tensor:
        states = batch["states"][index : index + 1].to(self.agent.device)
        actions = batch["actions"][index : index + 1].to(self.agent.device).flatten()

        if hasattr(self.agent, "actors"):
            actor, _ = self.agent._current_heads()
            logits = actor(self.agent.backbone(states / 255.0))
        else:
            logits = self.agent.actor(self.agent.network(states / 255.0))

        return -Categorical(logits=logits).log_prob(actions).mean()

    def compute_parameter_importance(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int = 100,
    ) -> dict[str, torch.Tensor]:
        """Estimate diagonal importance from per-sample squared gradients.

        PPO uses a policy empirical Fisher. DQN uses a squared TD-loss-gradient
        surrogate because a Q function does not define a policy likelihood.
        """
        if "states" not in batch or "actions" not in batch:
            raise ValueError("Importance estimation requires states and actions")
        if len(batch["states"]) == 0:
            raise ValueError("Importance estimation requires a non-empty batch")
        if num_samples <= 0:
            raise ValueError("Importance estimation requires at least one sample")

        params = self._collect_regularized_params()
        fisher = {name: torch.zeros_like(parameter) for name, parameter in params.items()}
        sample_count = min(num_samples, len(batch["states"]))
        indices = torch.randperm(len(batch["states"]))[:sample_count].tolist()
        is_dqn = "next_states" in batch

        for index in indices:
            self.agent.optimizer.zero_grad(set_to_none=True)
            loss = (
                self._dqn_sample_loss(batch, index)
                if is_dqn
                else self._ppo_sample_loss(batch, index)
            )
            loss.backward()
            for name, parameter in params.items():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().square())

        self.agent.optimizer.zero_grad(set_to_none=True)
        for value in fisher.values():
            value.div_(sample_count)
        return fisher

    def consolidate_weights(self, batch: dict[str, torch.Tensor] | None = None) -> dict:
        """Save a task snapshot and its empirical Fisher."""
        if batch is None:
            raise ValueError("EWC consolidation requires a representative task batch")

        params = self._collect_regularized_params()
        if not params:
            raise RuntimeError("Agent has no trainable parameters to consolidate")

        weights = {name: parameter.detach().clone() for name, parameter in params.items()}
        fisher = self.compute_parameter_importance(batch)
        module_summary = summarize_parameter_importance(fisher)
        task_id = self.current_task_id
        diagnostic = {
            "estimator": (
                "td_mse_squared_gradient_importance"
                if "next_states" in batch
                else "policy_nll_empirical_fisher"
            ),
            "is_empirical_fisher": "next_states" not in batch,
            "sample_count": min(100, len(batch["states"])),
            "protected_modules": list(module_summary),
            "modules": module_summary,
        }
        self.task_weights[task_id] = weights
        self.task_fisher[task_id] = fisher
        self.task_fisher_summary[task_id] = diagnostic
        self.current_task_id += 1
        return diagnostic

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute the EWC penalty over all consolidated tasks."""
        current_params = self._collect_regularized_params()
        penalty = torch.zeros((), device=self.agent.device)

        for task_id, previous_params in self.task_weights.items():
            fisher = self.task_fisher[task_id]
            for name, parameter in current_params.items():
                if name in previous_params:
                    penalty = (
                        penalty
                        + (fisher[name] * (parameter - previous_params[name]).square()).sum()
                    )

        return 0.5 * self.ewc_lambda * penalty

    def update(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> dict[str, float]:
        """Delegate the update while injecting the EWC penalty."""
        metrics = self.agent.update(
            batch,
            *args,
            regularizer=self.compute_ewc_loss,
            **kwargs,
        )
        regularization_loss = metrics.pop("regularization_loss", None)
        if regularization_loss is not None:
            metrics["ewc_loss"] = regularization_loss
        return metrics

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select an action through the wrapped agent."""
        return self.agent.select_action(state, deterministic=deterministic)

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor | None = None):
        """Delegate PPO action/value computation."""
        return self.agent.get_action_and_value(state, action)

    def get_value(self, state: torch.Tensor):
        """Delegate PPO value computation."""
        return self.agent.get_value(state)

    @property
    def device(self):
        """Return the wrapped agent's device."""
        return self.agent.device

    @property
    def gamma(self):
        """Return the wrapped agent's discount factor."""
        return self.agent.gamma

    def register_task(self, task_id: str, action_dim: int) -> None:
        """Register a task on a multi-head agent."""
        self.agent.register_task(task_id, action_dim)

    def set_task(self, task_id: str) -> None:
        """Select a task on a multi-head agent."""
        self.agent.set_task(task_id)

    def save(self, path: str) -> None:
        """Save both the wrapped agent and EWC consolidation state."""
        torch.save(
            {
                "format": "ewc-v1",
                "agent": self.agent.checkpoint_state(),
                "ewc_lambda": self.ewc_lambda,
                "task_weights": self.task_weights,
                "task_fisher": self.task_fisher,
                "task_fisher_summary": self.task_fisher_summary,
                "current_task_id": self.current_task_id,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore an EWC checkpoint, accepting old agent-only checkpoints."""
        checkpoint = safe_torch_load(path, map_location=self.device)
        if checkpoint.get("format") != "ewc-v1":
            self.agent.load_checkpoint_state(checkpoint)
            return

        self.agent.load_checkpoint_state(checkpoint["agent"])
        self.ewc_lambda = checkpoint["ewc_lambda"]
        self.task_weights = checkpoint["task_weights"]
        self.task_fisher = checkpoint["task_fisher"]
        self.task_fisher_summary = checkpoint.get("task_fisher_summary", {})
        self.current_task_id = checkpoint["current_task_id"]
