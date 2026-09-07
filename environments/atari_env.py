"""Atari environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode

from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_atari_env(
    game_name: str,
    frame_stack: int = 4,
    render_mode: str | None = None,
    seed: int | None = None,
    training: bool = True,
    frame_skip: int = 4,
) -> gym.Env:
    """Build the Gymnasium environment used by scalar and vector wrappers."""
    gym.register_envs(ale_py)
    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    if frame_skip <= 0:
        raise ValueError("frame_skip must be positive")

    if not game_name.startswith("ALE/"):
        game_name = f"ALE/{game_name}"

    make_kwargs = {"frameskip": 1}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    env = gym.make(game_name, **make_kwargs)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    if frame_skip > 1:
        env = MaxAndSkipEnv(env, skip=frame_skip)
    if training:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if training:
        env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)

    if seed is not None:
        env.action_space.seed(seed)
    return env


@dataclass(frozen=True)
class VectorStep:
    """One batched transition with post-reset and terminal observations separated."""

    observations: torch.Tensor
    transition_observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class AtariEnv:
    """Wrapper for Atari environments with cleanrl-style preprocessing."""

    def __init__(
        self,
        game_name: str,
        frame_stack: int = 4,
        render_mode: str | None = None,
        seed: int | None = None,
        training: bool = True,
        frame_skip: int = 4,
    ) -> None:
        """
        Initialize Atari environment with cleanrl-style wrappers.

        Args:
            game_name: Name of the Atari game (e.g., 'Pong-v5' or 'ALE/Pong-v5')
            frame_stack: Number of frames to stack (should be 4)
            render_mode: Render mode ('rgb_array' for visualization, None for training)
            seed: Random seed for environment
            training: Apply training-only life termination and reward clipping
            frame_skip: Number of emulator frames per agent action
        """
        self.game_name = game_name
        self.frame_stack = frame_stack
        self.training = training
        self.frame_skip = frame_skip
        self._reset_seed = seed

        self.env = make_atari_env(
            game_name,
            frame_stack=frame_stack,
            render_mode=render_mode,
            seed=seed,
            training=training,
            frame_skip=frame_skip,
        )

        self.action_space = self.env.action_space.n

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial state."""
        obs, _ = self.env.reset(seed=self._reset_seed)
        self._reset_seed = None
        return torch.as_tensor(np.asarray(obs))

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool]:
        """Take a step in the environment.

        Args:
            action: Discrete action index

        Returns:
            state: Current state
            reward: Reward
            terminated: Whether the transition reached an MDP terminal state
            truncated: Whether an external time limit ended the episode
        """
        if not self.env.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.env.action_space}")
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return torch.as_tensor(np.asarray(obs)), float(reward), terminated, truncated

    def close(self) -> None:
        """Close environment."""
        self.env.close()


class SyncVectorAtariEnv:
    """Step independent Atari environments synchronously for batched policies."""

    def __init__(
        self,
        game_name: str,
        num_envs: int,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
        training: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [
            AtariEnv(
                game_name,
                render_mode=render_mode,
                seed=None if seed is None else seed + index,
                training=training,
            )
            for index in range(num_envs)
        ]
        self.num_envs = num_envs
        self.action_space = self.envs[0].action_space

    def reset(self) -> torch.Tensor:
        """Reset every environment and stack observations on the leading axis."""
        return torch.stack([env.reset() for env in self.envs])

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step every environment once without automatically resetting finished ones."""
        actions = torch.as_tensor(actions, dtype=torch.long).flatten().cpu()
        if actions.numel() != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, received {actions.numel()}")

        transitions = [env.step(action.item()) for env, action in zip(self.envs, actions)]
        states, rewards, terminated, truncated = zip(*transitions)
        return (
            torch.stack(states),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
        )

    def reset_done(self, states: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Reset only completed environments while preserving other next states."""
        done = torch.as_tensor(done, dtype=torch.bool).flatten().cpu()
        if states.shape[0] != self.num_envs or done.numel() != self.num_envs:
            raise ValueError("State and done batches must match num_envs")
        if not done.any():
            return states

        states = states.clone()
        for index in done.nonzero().flatten().tolist():
            states[index] = self.envs[index].reset()
        return states

    def step_and_reset(self, actions: torch.Tensor) -> VectorStep:
        """Step once and reset completed environments without losing terminal frames."""
        transition_observations, rewards, terminated, truncated = self.step(actions)
        observations = self.reset_done(transition_observations, terminated | truncated)
        return VectorStep(
            observations=observations,
            transition_observations=transition_observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        """Close every underlying environment."""
        for env in self.envs:
            env.close()


class AsyncVectorAtariEnv:
    """Run independent Atari environments in shared-memory worker processes."""

    def __init__(
        self,
        game_name: str,
        num_envs: int,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
        training: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        env_fns = [
            partial(
                make_atari_env,
                game_name,
                render_mode=render_mode,
                seed=None if seed is None else seed + index,
                training=training,
            )
            for index in range(num_envs)
        ]
        self.env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=True,
            copy=False,
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        self.num_envs = num_envs
        self.action_space = self.env.single_action_space.n
        self._reset_seeds = None if seed is None else [seed + index for index in range(num_envs)]

    def reset(self) -> torch.Tensor:
        """Reset every worker using the deterministic per-environment seed once."""
        observations, _ = self.env.reset(seed=self._reset_seeds)
        self._reset_seeds = None
        return torch.as_tensor(np.asarray(observations))

    def step_and_reset(self, actions: torch.Tensor) -> VectorStep:
        """Step workers and recover final observations hidden by same-step autoreset."""
        actions = torch.as_tensor(actions, dtype=torch.long).flatten().cpu()
        if actions.numel() != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, received {actions.numel()}")

        observations, rewards, terminated, truncated, infos = self.env.step(actions.numpy())
        observations = np.asarray(observations)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)
        done = terminated | truncated
        transition_observations = observations

        if done.any():
            transition_observations = observations.copy()
            final_observations = infos.get("final_obs")
            final_mask = np.asarray(
                infos.get("_final_obs", np.zeros(self.num_envs, dtype=np.bool_)),
                dtype=np.bool_,
            )
            if final_observations is None or not np.array_equal(final_mask, done):
                raise RuntimeError(
                    "Async vector environment did not preserve every final observation"
                )
            for index in done.nonzero()[0]:
                transition_observations[index] = np.asarray(final_observations[index])

        return VectorStep(
            observations=torch.as_tensor(observations),
            transition_observations=torch.as_tensor(transition_observations),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            terminated=torch.as_tensor(terminated, dtype=torch.bool),
            truncated=torch.as_tensor(truncated, dtype=torch.bool),
        )

    def close(self) -> None:
        """Close every worker process."""
        self.env.close()


def make_vector_atari_env(
    game_name: str,
    num_envs: int,
    *,
    backend: str = "sync",
    render_mode: str | None = None,
    seed: int | None = None,
    training: bool = True,
) -> SyncVectorAtariEnv | AsyncVectorAtariEnv:
    """Build a vector environment while keeping the backend choice explicit."""
    environment_types = {
        "sync": SyncVectorAtariEnv,
        "async": AsyncVectorAtariEnv,
    }
    try:
        environment_type = environment_types[backend]
    except KeyError as error:
        raise ValueError(f"Unknown vector environment backend: {backend!r}") from error
    return environment_type(
        game_name,
        num_envs,
        render_mode=render_mode,
        seed=seed,
        training=training,
    )
