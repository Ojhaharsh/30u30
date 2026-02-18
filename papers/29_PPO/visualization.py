"""
visualization.py - Visualization utilities for PPO (Day 29)

Plots:
    1. Learning curves (reward over training)
    2. Clipping frequency over training (diagnostic for epsilon tuning)
    3. Policy entropy over training (should decrease as policy converges)
    4. Probability ratio distribution (shows how far policy moves each update)

Reference: Schulman et al. (2017) - https://arxiv.org/abs/1707.06347
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_learning_curve(
    rewards: List[float],
    title: str = "PPO Training: CartPole-v1",
    window: int = 20,
    save_path: str = None,
):
    """
    Plot episode rewards over training with a rolling average.

    Args:
        rewards: List of mean episode rewards per iteration.
        title: Plot title.
        window: Rolling average window size.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    iterations = np.arange(1, len(rewards) + 1)
    ax.plot(iterations, rewards, alpha=0.4, color="steelblue", label="Per-iteration reward")

    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, len(rewards) + 1),
            rolling,
            color="steelblue",
            linewidth=2,
            label=f"{window}-iteration rolling average",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_diagnostics(
    stats_history: List[Dict],
    save_path: str = None,
):
    """
    Plot PPO training diagnostics: clip fraction, entropy, value loss.

    The clip fraction is a key diagnostic for epsilon tuning:
    - Too low (< 5%): epsilon is too small, updates are too conservative
    - Too high (> 50%): epsilon is too large, updates are too aggressive
    - Healthy range: 5-30%

    Args:
        stats_history: List of dicts from PPOAgent.update(), one per iteration.
        save_path: If provided, save the figure to this path.
    """
    iterations = np.arange(1, len(stats_history) + 1)

    clip_fracs = [s["clip_fraction"] for s in stats_history]
    entropies = [s["entropy"] for s in stats_history]
    value_losses = [s["value_loss"] for s in stats_history]
    policy_losses = [s["policy_loss"] for s in stats_history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training Diagnostics", fontsize=14)

    # Clip fraction
    axes[0, 0].plot(iterations, clip_fracs, color="coral")
    axes[0, 0].axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="5% lower bound")
    axes[0, 0].axhline(y=0.30, color="gray", linestyle="--", alpha=0.5, label="30% upper bound")
    axes[0, 0].set_title("Clip Fraction")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Fraction of steps clipped")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Policy entropy
    axes[0, 1].plot(iterations, entropies, color="mediumseagreen")
    axes[0, 1].set_title("Policy Entropy")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Entropy (nats)")
    axes[0, 1].grid(True, alpha=0.3)

    # Value loss
    axes[1, 0].plot(iterations, value_losses, color="mediumpurple")
    axes[1, 0].set_title("Value Function Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Policy loss
    axes[1, 1].plot(iterations, policy_losses, color="steelblue")
    axes[1, 1].set_title("Policy Loss (L_CLIP)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_clipping_analysis(
    epsilon_values: List[float],
    rewards_per_epsilon: List[List[float]],
    save_path: str = None,
):
    """
    Plot final performance vs. epsilon (clip range).

    Reproduces the spirit of Table 1 in the paper (ablation on clipping).
    The paper uses epsilon=0.2 as the default.

    Args:
        epsilon_values: List of epsilon values tested.
        rewards_per_epsilon: List of reward histories, one per epsilon value.
        save_path: If provided, save the figure to this path.
    """
    final_rewards = [np.mean(r[-20:]) for r in rewards_per_epsilon]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [str(e) for e in epsilon_values],
        final_rewards,
        color="steelblue",
        alpha=0.8,
    )
    ax.axvline(
        x=epsilon_values.index(0.2) if 0.2 in epsilon_values else -1,
        color="coral",
        linestyle="--",
        label="Paper default (epsilon=0.2)",
    )
    ax.set_xlabel("Epsilon (clip range)")
    ax.set_ylabel("Final mean reward (last 20 iterations)")
    ax.set_title("Performance vs. Clip Range (Ablation)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Demo: plot synthetic data to verify the visualization functions work
    np.random.seed(42)
    n_iter = 100

    # Synthetic reward curve (CartPole-like: starts low, converges to ~475)
    rewards = [
        min(475, 50 + i * 4 + np.random.randn() * 20) for i in range(n_iter)
    ]

    # Synthetic stats
    stats_history = [
        {
            "clip_fraction": max(0, 0.15 - i * 0.001 + np.random.randn() * 0.02),
            "entropy": max(0.1, 0.7 - i * 0.005 + np.random.randn() * 0.02),
            "value_loss": max(0, 50 - i * 0.4 + np.random.randn() * 5),
            "policy_loss": np.random.randn() * 0.01,
        }
        for i in range(n_iter)
    ]

    plot_learning_curve(rewards, title="PPO Demo: CartPole-v1 (synthetic)")
    plot_training_diagnostics(stats_history)
    print("Visualization demo complete.")
