"""Train a single agent on a single Atari game."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from algorithms import (
    DEFAULT_DQN_LEARNING_STARTS,
    DQNAgent,
    PPOAgent,
)
from environments import AtariEnv, SyncVectorAtariEnv
from utils import MetricsPlotter, ReplayBuffer, RolloutBuffer, VideoRecorder, seed_everything


def train_single_game(
    game_name: str = "Pong-v5",
    algorithm: str = "dqn",
    num_steps: int = 500000,
    batch_size: int = 32,
    num_envs: int = 1,
    save_video: bool = False,
    seed: int = 0,
):
    """Train agent on a single game."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if algorithm != "ppo" and num_envs != 1:
        raise ValueError("num_envs greater than one is currently supported only for PPO")
    if algorithm == "ppo" and num_steps % num_envs != 0:
        raise ValueError("PPO steps must be divisible by num_envs")
    seed_everything(seed)
    print(f"Training {algorithm.upper()} on {game_name}")

    env = (
        SyncVectorAtariEnv(game_name, num_envs, render_mode=None, seed=seed)
        if algorithm == "ppo"
        else AtariEnv(game_name, render_mode=None, seed=seed)
    )

    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_fraction=0.10,
            total_timesteps=num_steps,
            target_update_freq=1000,
            tau=1.0,
        )
        learning_starts = DEFAULT_DQN_LEARNING_STARTS
        train_frequency = 4
        buffer = ReplayBuffer(capacity=100000)
    else:
        agent = PPOAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        learning_starts = 0
        train_frequency = 1
        rollout_length = 128
        update_epochs = 4
        minibatch_size = batch_size
        buffer = RolloutBuffer(capacity=rollout_length, num_envs=num_envs)

    run_name = f"{game_name}_{algorithm}"
    exp_dir = Path("outputs") / "single" / run_name / f"seed-{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = None
    if save_video:
        video_path = exp_dir / "training.mp4"
        video_recorder = VideoRecorder(
            str(video_path),
            fps=30,
        )

    metrics_plotter = MetricsPlotter()
    episode_rewards = []

    state = env.reset()
    episode_reward = [0.0] * num_envs if algorithm == "ppo" else 0.0
    episode_count = 0

    pbar = tqdm(total=num_steps, desc="Training")
    step = 0

    while step < num_steps:
        if algorithm == "dqn":
            agent.global_step = step
            action = agent.select_action(state)
            next_state, reward, terminated, truncated = env.step(action)
            episode_done = terminated or truncated
            episode_reward += reward
            buffer.add(state, action, reward, next_state, terminated)

            if video_recorder is not None and step % 2 == 0:
                frame = state[0].cpu().numpy() if isinstance(state, torch.Tensor) else state[0]
                video_recorder.add_frame(frame)

            if step >= learning_starts and step % train_frequency == 0:
                if buffer.is_ready(batch_size):
                    batch = buffer.sample(batch_size)
                    metrics = agent.update(batch)
                    pbar.set_postfix(metrics)

                    if "loss" in metrics:
                        metrics_plotter.add_metric("loss", metrics["loss"])
                    if "epsilon" in metrics:
                        metrics_plotter.add_metric("epsilon", metrics["epsilon"])
                    if "q_value" in metrics:
                        metrics_plotter.add_metric("q_value", metrics["q_value"])

            state = next_state
            step += 1
            pbar.update(1)

            if episode_done:
                episode_rewards.append(episode_reward)
                metrics_plotter.add_metric("episode_reward", episode_reward)
                episode_reward = 0
                state = env.reset()

        else:
            collected_steps = 0
            for rollout_step in range(rollout_length):
                if step >= num_steps:
                    break

                with torch.inference_mode():
                    state_tensor = state.to(agent.device)
                    action_tensor, log_prob, value = agent.sample_action_and_value(state_tensor)
                actions = action_tensor.cpu()

                next_state, rewards, terminated, truncated = env.step(actions)
                episode_done = terminated | truncated
                training_rewards = rewards.clone()
                bootstrap_mask = truncated & ~terminated
                if bootstrap_mask.any():
                    with torch.inference_mode():
                        truncated_values = agent.get_value(
                            next_state[bootstrap_mask].to(agent.device)
                        ).flatten()
                    training_rewards[bootstrap_mask] += agent.gamma * truncated_values.cpu()

                if num_envs == 1:
                    buffer.add(
                        state[0],
                        actions.item(),
                        training_rewards.item(),
                        episode_done.item(),
                        log_prob.item(),
                        value.item(),
                    )
                else:
                    buffer.add(
                        state,
                        actions,
                        training_rewards,
                        episode_done,
                        log_prob.cpu(),
                        value.flatten().cpu(),
                    )

                if video_recorder is not None and step % 2 == 0:
                    frame = state[0, 0].cpu().numpy()
                    video_recorder.add_frame(frame)

                for env_index in range(num_envs):
                    episode_reward[env_index] += rewards[env_index].item()
                    if episode_done[env_index]:
                        reward_value = episode_reward[env_index]
                        episode_rewards.append(reward_value)
                        metrics_plotter.add_metric("episode_reward", reward_value)
                        episode_reward[env_index] = 0.0
                        episode_count += 1

                state = env.reset_done(next_state, episode_done)
                step += num_envs
                collected_steps += num_envs

            if buffer.ready_for_update(final=step >= num_steps):
                with torch.inference_mode():
                    next_value = agent.get_value(state.to(agent.device)).flatten()

                rollout_data = buffer.get_batch()
                metrics = agent.update(rollout_data, next_value, update_epochs, minibatch_size)
                pbar.set_postfix({**metrics, "episodes": episode_count})

                if "policy_loss" in metrics:
                    metrics_plotter.add_metric("policy_loss", metrics["policy_loss"])
                if "value_loss" in metrics:
                    metrics_plotter.add_metric("value_loss", metrics["value_loss"])
                if "entropy" in metrics:
                    metrics_plotter.add_metric("entropy", metrics["entropy"])

                buffer.reset()

            pbar.update(collected_steps)

    if video_recorder is not None:
        video_recorder.save(format="mp4")
        print(f"Video saved to {exp_dir / 'training.mp4'}")

    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        last_n = min(20, len(episode_rewards))
        avg_last_n = sum(episode_rewards[-last_n:]) / last_n
        print(f"\n{'=' * 60}")
        print("Training Summary")
        print(f"{'=' * 60}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average episode reward: {avg_reward:.2f}")
        print(f"Average reward over last {last_n} episodes: {avg_last_n:.2f}")

        if len(episode_rewards) >= 20:
            first_10 = sum(episode_rewards[:10]) / 10
            last_10 = sum(episode_rewards[-10:]) / 10
            print("\nReward Progression:")
            print(f"  First 10 episodes: {first_10:.2f}")
            print(f"  Last 10 episodes: {last_10:.2f}")
            improvement = last_10 - first_10
            print(f"  Improvement: {improvement:+.2f}")
            if improvement > 0:
                print("  ✓ Agent is learning!")
            elif improvement < -1:
                print("  ⚠ Agent performance decreased")
            else:
                print("  → Agent performance stable")

        best_ep = max(episode_rewards)
        worst_ep = min(episode_rewards)
        print(f"\nBest episode reward: {best_ep:.2f}")
        print(f"Worst episode reward: {worst_ep:.2f}")
        print(f"{'=' * 60}")

    metrics_output_path = exp_dir / "metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Metrics plot saved to {metrics_output_path}")

    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = Path("checkpoints") / "single" / run_name / f"seed-{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))
    print(f"Agent saved to {checkpoint_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pong-v5", help="Game name")
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-video", action="store_true", help="Enable video recording (disabled by default)"
    )

    args = parser.parse_args()

    train_single_game(
        game_name=args.game,
        algorithm=args.algorithm,
        num_steps=args.steps,
        batch_size=args.batch_size,
        num_envs=args.num_envs,
        save_video=args.save_video,
        seed=args.seed,
    )
