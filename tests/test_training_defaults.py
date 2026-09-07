"""Regression tests for teaching-scale training defaults."""

from inspect import signature

import pytest

from algorithms import DEFAULT_DQN_LEARNING_STARTS
from scripts.train_continual import train_continual
from scripts.train_multitask import train_multitask
from scripts.train_single import train_single_game


def test_continual_dqn_starts_learning_before_default_task_ends() -> None:
    default_task_steps = signature(train_continual).parameters["steps_per_game"].default

    assert DEFAULT_DQN_LEARNING_STARTS < default_task_steps


def test_continual_training_has_a_reproducible_default_seed() -> None:
    assert signature(train_continual).parameters["seed"].default == 0


@pytest.mark.parametrize(
    "train",
    (train_single_game, train_continual, train_multitask),
)
def test_training_entry_points_reject_invalid_batch_size(train) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        train(batch_size=0)
