"""
visualization.py - Results Visualization for Day 30 (RLHF)

This script plots the training metrics from `train_minimal.py`:
1.  Reward Evolution: Comparing the "True" environment reward vs. the "Learned" Reward Model output.
2.  Reward Model Accuracy: How often the RM agrees with the Synthetic Oracle.

References:
    Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"
    https://arxiv.org/abs/1706.03741

Usage:
    python visualization.py
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize():
    if not os.path.exists('rlhf_results.pt'):
        print("Error: 'rlhf_results.pt' not found.")
        print("Run training first:\n    python train_minimal.py --steps 5000")
        return

    try:
        results = torch.load('rlhf_results.pt')
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    true_r = results['true_rewards']
    learned_r = results['learned_rewards']
    accuracies = results['rm_accuracies']
    
    # Validation check
    if len(true_r) == 0:
        print("Results file is empty. Train for more steps.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Reward Evolution
    ax1.set_title("Reward Alignment Breakdown")
    ax1.set_xlabel("RLHF Cycle")
    ax1.set_ylabel("True Environment Reward", color='tab:blue', fontweight='bold')
    l1 = ax1.plot(true_r, label="True Reward (Ground Truth)", color='tab:blue', marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    
    ax1_right = ax1.twinx()
    ax1_right.set_ylabel("Learned Reward Model Output", color='tab:orange', fontweight='bold')
    l2 = ax1_right.plot(learned_r, label="Learned Reward (Proxy)", color='tab:orange', marker='x', linestyle='--', linewidth=2)
    ax1_right.tick_params(axis='y', labelcolor='tab:orange')
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 2. Accuracy
    ax2.set_title("Reward Model Classification Accuracy")
    ax2.set_xlabel("RLHF Cycle")
    ax2.set_ylabel("Pairwise Accuracy")
    ax2.plot(accuracies, color='green', marker='s', linewidth=2, label="Validation Accuracy")
    ax2.set_ylim(0.4, 1.0)
    ax2.axhline(0.5, color='gray', linestyle='--', label="Random Guessing (0.5)")
    ax2.axhline(1.0, color='gray', linestyle=':', label="Perfect (1.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Day 30: Deep RL from Human Feedback (Christiano et al. 2017)", fontsize=14)
    plt.tight_layout()
    
    save_path = 'rlhf_training.png'
    plt.savefig(save_path, dpi=150)
    print(f"Success! Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    print("-" * 50)
    print("Day 30 Visualization Suite")
    print("-" * 50)
    visualize()
