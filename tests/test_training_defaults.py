"""Regression tests for teaching-scale training defaults."""

from inspect import signature

from algorithms import DEFAULT_DQN_LEARNING_STARTS
from scripts.train_continual import train_continual


def test_continual_dqn_starts_learning_before_default_task_ends() -> None:
    default_task_steps = signature(train_continual).parameters["steps_per_game"].default

    assert DEFAULT_DQN_LEARNING_STARTS < default_task_steps
