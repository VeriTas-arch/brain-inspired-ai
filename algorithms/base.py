"""Base agent and shared network components."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


def safe_torch_load(path: str, map_location=None):
    """Load tensor-only checkpoints without allowing arbitrary object creation."""
    return torch.load(path, map_location=map_location, weights_only=True)


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, state_dim: int, action_dim: int, device: str = "cuda"):
        """
        Initialize the base agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Number of actions
            device: Device to use (cuda or cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.environment_protocol = "gymnasium_wrappers_v1"

        if device == "cuda" and not torch.cuda.is_available():
            import warnings

            warnings.warn("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.network = None

    @abstractmethod
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select an action given a state."""
        raise NotImplementedError

    def checkpoint_state(self) -> dict:
        """Return serializable state needed to restore the agent."""
        if self.network is None:
            return {}
        return {"network": self.network.state_dict()}

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore state produced by :meth:`checkpoint_state`."""
        if self.network is not None and "network" in checkpoint:
            self.network.load_state_dict(checkpoint["network"])

    def save(self, path: str):
        """Save the complete agent checkpoint."""
        torch.save(self.checkpoint_state(), path)

    def load(self, path: str):
        """Load an agent checkpoint."""
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.load_checkpoint_state(checkpoint)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SimpleNet(nn.Module):
    """Simple neural network for Atari."""

    def __init__(self, input_channels: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)


class AtariBackbone(nn.Module):
    """Shared convolutional backbone for Atari agents.

    Produces a feature vector (default 512-dim) from stacked frames.
    """

    def __init__(self, input_channels: int, feature_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, feature_dim),
            nn.ReLU(),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)
