"""
visualization.py - Visualization Suite for Universal Intelligence

Generates key plots demonstrating agent performance across the 
complexity spectrum and the universal weighting 2^-K.

Usage:
    python visualization.py

Reference: Shane Legg (2008) - http://www.vetta.org/documents/Machine_Super_Intelligence.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from implementation import (
    UniversalIntelligenceMeasure, 
    GridWorld, 
    PatternSequence, 
    RandomAgent, 
    SimpleRLAgent
)


def plot_intelligence_spectrum(results_dict, agent_name):
    """
    Visualize expected reward vs. environment complexity.
    
    This plot shows the 'Intelligence Spectrum' of an agent, illustrating
    its ability to solve problems of varying difficulty.
    
    Args:
        results_dict: Dictionary containing detailed benchmark results
        agent_name: Name of the agent being visualized
    """
    details = results_dict["details"]
    complexities = [d["complexity"] for d in details]
    rewards = [d["expected_reward"] for d in details]
    weights = [d["weight"] for d in details]
    env_names = [d["env"] for d in details]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Intelligence Spectrum
    ax1.scatter(complexities, rewards, color='skyblue', s=100, edgecolors='navy', alpha=0.7)
    for i, txt in enumerate(env_names):
        ax1.annotate(f"{txt}(K={complexities[i]})", (complexities[i], rewards[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')

    ax1.axhline(y=results_dict["upsilon_normalized"], color='r', linestyle='--', 
                label=f'Upsilon: {results_dict["upsilon_normalized"]:.4f}')
    ax1.set_title(f"Intelligence Spectrum: {agent_name}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Complexity K(mu)", fontsize=12)
    ax1.set_ylabel("Expected Reward V_mu", fontsize=12)
    ax1.set_ylim(-0.1, 1.2)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Plot 2: Universal Distribution (2^-K)
    # This shows why simple environments dominate the weighted average score.
    ax2.bar(range(len(complexities)), weights, color='salmon', alpha=0.7)
    ax2.set_xticks(range(len(complexities)))
    ax2.set_xticklabels(env_names, rotation=45)
    ax2.set_title("Universal Weighting (2^-K)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Contribution Weight", fontsize=12)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle("Day 27 | Universal Intelligence vs. Algorithmic Complexity", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("intelligence_spectrum.png", dpi=300)
    print(f"[OK] Visualization saved to intelligence_spectrum.png")


def analyze_agent_performance(envs, agent_types):
    """
    Print a comparative analysis of different agent architectures.
    """
    print("=" * 60)
    print("UNIVERSAL INTELLIGENCE ANALYSIS")
    print("=" * 60)
    
    measure = UniversalIntelligenceMeasure(envs)
    
    for name, agent_cls in agent_types.items():
        print(f"\nAnalyzing {name}...")
        results = measure.evaluate(agent_cls(), episodes=50)
        print(f"  Upsilon Score: {results['upsilon_normalized']:.4f}")
        
        # Detail breakdown
        print(f"  {'Env':<15} | {'K':<3} | {'Reward':<6}")
        for d in results['details']:
            print(f"  {d['env']:<15} | {d['complexity']:<3} | {d['expected_reward']:>6.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day 27: Universal Intelligence Visualization")
    print("=" * 60 + "\n")

    # 1. Setup Benchmark Suite
    envs = [
        GridWorld(size=3),
        GridWorld(size=5),
        GridWorld(size=8),
        PatternSequence([0, 1]),
        PatternSequence([0, 1, 0, 2]),
        PatternSequence([1, 2, 3, 4, 1, 2, 3, 4])
    ]

    # 2. Comparative Analysis
    agent_types = {
        "Random Baseline": RandomAgent,
        "Simple RL Agent": SimpleRLAgent
    }
    analyze_agent_performance(envs, agent_types)

    # 3. Generate Visual Spectrum
    print("\nGenerating final spectrum visualization...")
    rl_agent = SimpleRLAgent()
    measure = UniversalIntelligenceMeasure(envs)
    res_rl = measure.evaluate(rl_agent, episodes=50)
    plot_intelligence_spectrum(res_rl, "Simple RL Agent")
