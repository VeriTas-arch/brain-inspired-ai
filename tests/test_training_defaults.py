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


@pytest.mark.parametrize("train", (train_single_game, train_continual))
def test_ppo_vector_training_rejects_invalid_environment_counts(train) -> None:
    with pytest.raises(ValueError, match="num_envs must be positive"):
        train(algorithm="ppo", num_envs=0)


def test_ppo_vector_training_requires_divisible_step_budgets() -> None:
    with pytest.raises(ValueError, match="divisible"):
        train_single_game(algorithm="ppo", num_steps=10, num_envs=8)
    with pytest.raises(ValueError, match="divisible"):
        train_continual(algorithm="ppo", steps_per_game=10, num_envs=8)


@pytest.mark.parametrize("train", (train_single_game, train_continual))
def test_optimized_runtime_options_are_ppo_only(train) -> None:
    with pytest.raises(ValueError, match="only for PPO"):
        train(algorithm="dqn", env_backend="async")
    with pytest.raises(ValueError, match="only for PPO"):
        train(algorithm="dqn", compile_ppo=True)
