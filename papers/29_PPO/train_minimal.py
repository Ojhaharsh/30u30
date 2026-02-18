"""
train_minimal.py - PPO training script for Day 29

Trains a PPO-Clip agent on a gym environment.

Usage:
    python train_minimal.py --env CartPole-v1 --epochs 100
    python train_minimal.py --env LunarLander-v2 --epochs 300 --lr 3e-4
    python train_minimal.py --env MountainCar-v0 --epochs 500 --c2 0.01

Reference: Schulman et al. (2017) - https://arxiv.org/abs/1707.06347
"""

import argparse
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import torch
from implementation import PPOAgent


def train(args):
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f"Environment: {args.env}")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Hyperparameters: T={args.T}, K={args.K}, M={args.M}, "
          f"epsilon={args.epsilon}, lr={args.lr}, gamma={args.gamma}, "
          f"lambda={args.lambda_}, c1={args.c1}, c2={args.c2}")
    print("-" * 70)

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        T=args.T,
        K=args.K,
        M=args.M,
        epsilon=args.epsilon,
        gamma=args.gamma,
        lambda_=args.lambda_,
        lr=args.lr,
        c1=args.c1,
        c2=args.c2,
        max_grad_norm=args.max_grad_norm,
    )

    reward_history = []

    for iteration in range(args.epochs):
        mean_reward = agent.collect_rollout(env)
        stats = agent.update()
        reward_history.append(mean_reward)

        if (iteration + 1) % args.log_interval == 0:
            recent_mean = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else np.mean(reward_history)
            print(
                f"Epoch {iteration+1:4d}/{args.epochs} | "
                f"reward={mean_reward:7.1f} | "
                f"avg20={recent_mean:7.1f} | "
                f"clip_frac={stats['clip_fraction']:.3f} | "
                f"entropy={stats['entropy']:.4f}"
            )

    env.close()
    print()
    print(f"Final 20-episode average: {np.mean(reward_history[-20:]):.1f}")


def main():
    parser = argparse.ArgumentParser(description="PPO-Clip training script (Day 29)")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gym environment name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--T", type=int, default=2048,
                        help="Timesteps per iteration (paper default: 2048)")
    parser.add_argument("--K", type=int, default=10,
                        help="Epochs per iteration (paper default: 10)")
    parser.add_argument("--M", type=int, default=64,
                        help="Minibatch size (paper default: 64)")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Clip range (paper default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (paper default: 0.99)")
    parser.add_argument("--lambda_", type=float, default=0.95,
                        help="GAE lambda (paper default: 0.95)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (paper default: 3e-4)")
    parser.add_argument("--c1", type=float, default=1.0,
                        help="Value loss coefficient (paper default: 1.0)")
    parser.add_argument("--c2", type=float, default=0.01,
                        help="Entropy coefficient (paper: 0.01 Atari, 0 MuJoCo)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Gradient clipping max norm (paper default: 0.5)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print stats every N iterations")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
