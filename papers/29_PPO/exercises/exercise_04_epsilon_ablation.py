"""
Exercise 4: Epsilon Ablation

Train PPO with different epsilon (clip range) values and compare performance.

This reproduces the spirit of Table 1 in the paper (Section 6.1), which shows
that the clipping mechanism is the key component of PPO's performance.

The paper uses epsilon=0.2 as the default. This exercise tests:
    epsilon in {0.05, 0.1, 0.2, 0.3, 0.5}

Reference: Schulman et al. (2017), Table 1 and Section 6.1
https://arxiv.org/abs/1707.06347
"""

import numpy as np
import sys
import os

# Add parent directory to path to import implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import PPOAgent

try:
    import gymnasium as gym
except ImportError:
    import gym


def run_epsilon_ablation(
    env_name: str = "CartPole-v1",
    n_iterations: int = 50,
    epsilon_values: list = None,
    seed: int = 42,
) -> dict:
    """
    Train PPO with different epsilon values and return final performance.

    Args:
        env_name: Gym environment name.
        n_iterations: Training iterations per epsilon value.
        epsilon_values: List of epsilon values to test.
        seed: Random seed for reproducibility.

    Returns:
        results: Dict mapping epsilon -> list of mean rewards per iteration.
    """
    if epsilon_values is None:
        epsilon_values = [0.05, 0.1, 0.2, 0.3, 0.5]

    results = {}

    for epsilon in epsilon_values:
        print(f"Training with epsilon={epsilon}...")
        np.random.seed(seed)
        torch_seed = seed
        try:
            import torch
            torch.manual_seed(torch_seed)
        except ImportError:
            pass

        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        agent = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            T=512,
            K=4,
            M=64,
            epsilon=epsilon,
            gamma=0.99,
            lambda_=0.95,
            lr=3e-4,
            c1=1.0,
            c2=0.01,
        )

        rewards = []
        for iteration in range(n_iterations):
            mean_reward = agent.collect_rollout(env)
            agent.update()
            rewards.append(mean_reward)

        env.close()
        final_avg = np.mean(rewards[-10:])
        print(f"  epsilon={epsilon}: final 10-iter avg = {final_avg:.1f}")
        results[epsilon] = rewards

    return results


def analyze_results(results: dict):
    """
    Print a summary table of results, similar to Table 1 in the paper.

    Args:
        results: Dict from run_epsilon_ablation.
    """
    print()
    print("Epsilon Ablation Results (spirit of Table 1 in the paper)")
    print("-" * 50)
    print(f"{'Epsilon':>10} | {'Final Avg Reward':>18} | {'Max Reward':>12}")
    print("-" * 50)

    for epsilon, rewards in sorted(results.items()):
        final_avg = np.mean(rewards[-10:])
        max_reward = max(rewards)
        marker = " <-- paper default" if epsilon == 0.2 else ""
        print(f"{epsilon:>10.2f} | {final_avg:>18.1f} | {max_reward:>12.1f}{marker}")

    print()
    print("Expected finding: epsilon=0.2 should perform best or near-best.")
    print("Very small epsilon (0.05) = too conservative, slow learning.")
    print("Very large epsilon (0.5) = too aggressive, unstable training.")


if __name__ == "__main__":
    print("Exercise 4: Epsilon Ablation")
    print("This will train PPO 5 times (once per epsilon value).")
    print("Expected runtime: ~2-5 minutes on CPU.")
    print()

    results = run_epsilon_ablation(
        env_name="CartPole-v1",
        n_iterations=50,
        epsilon_values=[0.05, 0.1, 0.2, 0.3, 0.5],
    )
    analyze_results(results)

    # Optional: plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from visualization import plot_clipping_analysis

        plot_clipping_analysis(
            epsilon_values=list(results.keys()),
            rewards_per_epsilon=list(results.values()),
        )
    except ImportError:
        print("matplotlib not available, skipping plot.")
