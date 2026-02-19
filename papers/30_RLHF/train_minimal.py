"""
train_minimal.py - Training Script for Day 30 (RLHF)

This script implements the full RLHF loop:
1.  **Collect**: Run the policy in the environment to gather trajectory segments.
2.  **Label**: Use the Synthetic Oracle to label pairs of segments.
3.  **Train Reward Model**: Update the reward network to predict the Oracle's preferences.
4.  **Train Policy**: Optimize the PPO agent using the *learned* reward.

Reference:
    Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"
    https://arxiv.org/abs/1706.03741

Usage:
    python train_minimal.py --env CartPole-v1 --steps 5000
    python train_minimal.py --env CartPole-v1 --segment_len 25 --pairs 50
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from implementation import RLHF_Trainer, RewardModel
import os

try:
    import gymnasium as gym
except ImportError:
    import gym

def train(env_name, total_steps, segment_length, pairs_per_batch, rm_epochs, ppo_epochs):
    print(f"Training RLHF on {env_name}...")
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    trainer = RLHF_Trainer(env, obs_dim, act_dim, segment_length=segment_length)
    
    # Tracking
    true_rewards = []
    learned_rewards = []
    rm_accuracies = []
    
    # Main Loop
    # We do N cycles. Each cycle:
    # 1. Collect trajectories & Get Labels (Synthetic Oracle)
    # 2. Train Reward Model
    # 3. Train Policy (PPO) using Learned Reward
    
    cycles = total_steps // (segment_length * pairs_per_batch * 2) # Rough estimate
    cycles = max(1, cycles) # At least 1 cycle
    
    print(f"Starting {cycles} RLHF cycles.")
    
    for cycle in range(cycles):
        print(f"\n--- Cycle {cycle+1}/{cycles} ---")
        
        # 1. Collect & Label
        print("Collecting preferences...")
        trainer.collect_and_label_data(num_new_pairs=pairs_per_batch)
        print(f"Buffer size: {len(trainer.preference_buffer)}")
        
        # 2. Train Reward Model
        print("Training Reward Model...")
        rm_loss, rm_acc = trainer.train_reward_model(epochs=rm_epochs)
        rm_accuracies.append(rm_acc)
        print(f"RM Loss: {rm_loss:.4f} | RM Accuracy: {rm_acc:.2%}")
        
        # 3. Train Policy
        print("Optimizing Policy (PPO)...")
        mean_learned_reward = trainer.train_policy_ppo(steps=ppo_epochs * 200) # Arbitrary steps per cycle
        learned_rewards.append(mean_learned_reward)
        
        # Evaluate Policy (True Reward)
        eval_rewards = []
        for _ in range(5):
            obs, _ = env.reset()
            done = False
            total_r = 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, _, _, _ = trainer.policy.get_action_and_value(obs_t)
                obs, r, done, trunc, _ = env.step(action.item())
                total_r += r
                if done or trunc: break
            eval_rewards.append(total_r)
        
        mean_true = np.mean(eval_rewards)
        true_rewards.append(mean_true)
        print(f"Policy True Reward: {mean_true:.1f} (Learned: {mean_learned_reward:.3f})")
        
    env.close()
    
    # Save results
    results = {
        'true_rewards': true_rewards,
        'learned_rewards': learned_rewards,
        'rm_accuracies': rm_accuracies
    }
    torch.save(results, 'rlhf_results.pt')
    torch.save(trainer.reward_model.state_dict(), 'reward_model.pth')
    print("\nTraining complete. Results saved to rlhf_results.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF Training Script (Day 30)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment ID")
    parser.add_argument("--steps", type=int, default=10000, help="Total environment steps roughly")
    parser.add_argument("--segment_len", type=int, default=50, help="Length of video clips (dataset generation)")
    parser.add_argument("--pairs", type=int, default=10, help="Number of preference pairs per cycle")
    parser.add_argument("--rm_epochs", type=int, default=5, help="Epochs to train RM per cycle")
    parser.add_argument("--ppo_epochs", type=int, default=5, help="PPO updates per cycle")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Day 30: RLHF Training on {args.env}")
    print("=" * 50)
    
    train(args.env, args.steps, args.segment_len, args.pairs, args.rm_epochs, args.ppo_epochs)
