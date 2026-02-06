"""
Solution 2: Interacting vs Non-Interacting Model Comparison
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import CoffeeAutomaton, measure_complexity, measure_entropy


def run_and_measure(model: str, grid_size: int = 50,
                    total_steps: int = 50000, num_snapshots: int = 50):
    """Run a simulation and collect measurements."""
    ca = CoffeeAutomaton(grid_size, model, seed=42)
    steps_per = total_steps // num_snapshots

    times = [0]
    complexities = [ca.complexity()]
    entropies = [ca.entropy()]

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)
        complexities.append(ca.complexity())
        entropies.append(ca.entropy())

    return times, complexities, entropies


def plot_comparison(times_inter, cx_inter, times_non, cx_non):
    """Plot complexity for both models."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.plot(times_inter, cx_inter, 'r-', label='Interacting', linewidth=2)
    plt.plot(times_non, cx_non, 'b--', label='Non-interacting', linewidth=2)

    plt.xlabel('Time step')
    plt.ylabel('Apparent complexity (gzip bytes)')
    plt.title('Interacting vs Non-Interacting Models\n'
              'Only the interacting model shows genuine complexity')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Running interacting model...")
    t_i, cx_i, e_i = run_and_measure('interacting')

    print("Running non-interacting model...")
    t_n, cx_n, e_n = run_and_measure('non_interacting')

    print(f"\nInteracting: peak complexity = {max(cx_i)}")
    print(f"Non-interacting: peak complexity = {max(cx_n)}")
    print(f"Ratio: {max(cx_i) / max(cx_n):.1f}x")

    plot_comparison(t_i, cx_i, t_n, cx_n)
