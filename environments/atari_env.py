"""Atari environment wrapper."""

from __future__ import annotations

import ale_py
import gymnasium as gym
import numpy as np
import torch

from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
        gym.register_envs(ale_py)
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")

        self.game_name = game_name
        self.frame_stack = frame_stack
        self.training = training
        self.frame_skip = frame_skip
        self._reset_seed = seed

        if not game_name.startswith("ALE/"):
            game_name = f"ALE/{game_name}"

        make_kwargs = {"frameskip": 1}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        self.env = gym.make(game_name, **make_kwargs)

        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = NoopResetEnv(self.env, noop_max=30)
        if frame_skip > 1:
            self.env = MaxAndSkipEnv(self.env, skip=frame_skip)
        if training:
            self.env = EpisodicLifeEnv(self.env)
        if "FIRE" in self.env.unwrapped.get_action_meanings():
            self.env = FireResetEnv(self.env)
        if training:
            self.env = ClipRewardEnv(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
        self.env = gym.wrappers.GrayscaleObservation(self.env)
        self.env = gym.wrappers.FrameStackObservation(self.env, frame_stack)

        if seed is not None:
            self.env.action_space.seed(seed)

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

    def close(self) -> None:
        """Close every underlying environment."""
        for env in self.envs:
            env.close()
