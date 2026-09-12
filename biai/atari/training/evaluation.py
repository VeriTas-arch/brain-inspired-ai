"""Shared deterministic evaluation boundaries."""

import torch

DEFAULT_MAX_EPISODE_STEPS = 30_000


def evaluation_schedule(total_steps, *, points=0, interval=0, step_size=1):
    """Place evaluations at completed updates, including the final training boundary."""
    if points < 0 or interval < 0 or (points and interval):
        raise ValueError("Use nonnegative eval_points or eval_interval, not both")
    if points:
        targets = [(total_steps * i + points - 1) // points for i in range(1, points + 1)]
    elif interval:
        targets = [*range(interval, total_steps, interval), total_steps]
    else:
        return []
    steps = sorted(
        {min(total_steps, ((t + step_size - 1) // step_size) * step_size) for t in targets}
    )
    if points and len(steps) != points:
        raise ValueError("eval_points exceeds the distinct training update boundaries")
    return steps


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
