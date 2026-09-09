"""Shared deterministic evaluation boundaries."""

import torch

DEFAULT_MAX_EPISODE_STEPS = 30_000


def run_evaluation_episodes(
    agent, environment, episodes: int, max_steps: int, *, frame_callback=None
) -> list[float]:
    """Return raw rewards from complete deterministic episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    rewards = []
    with torch.inference_mode():
        for episode_index in range(episodes):
            state = environment.reset()
            episode_reward = 0.0
            for _ in range(max_steps):
                if frame_callback is not None:
                    frame_callback()
                action = agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated = environment.step(action)
                episode_reward += reward
                if terminated or truncated:
                    rewards.append(episode_reward)
                    break
            else:
                raise RuntimeError(
                    f"Evaluation episode {episode_index + 1} did not finish within "
                    f"{max_steps} steps; increase the evaluation step limit instead of "
                    "recording a partial score"
                )
    return rewards
