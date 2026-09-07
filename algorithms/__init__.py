"""RL Algorithms module."""

from .base import BaseAgent, SimpleNet
from .dqn import DEFAULT_DQN_LEARNING_STARTS, DQNAgent, MultiHeadDQNAgent
from .ewc import EWCWrapper
from .ppo import MultiHeadPPOAgent, PPOAgent

__all__ = [
    "BaseAgent",
    "DEFAULT_DQN_LEARNING_STARTS",
    "SimpleNet",
    "DQNAgent",
    "MultiHeadDQNAgent",
    "PPOAgent",
    "MultiHeadPPOAgent",
    "EWCWrapper",
]
