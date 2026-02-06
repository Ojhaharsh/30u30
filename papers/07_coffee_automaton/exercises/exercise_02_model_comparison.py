"""
Exercise 2: Interacting vs Non-Interacting Model Comparison

The paper's key empirical finding (Section 6): genuine complexity requires
interaction. The non-interacting model never develops macroscopic complexity,
because each particle is an independent random walk.

Your task:
  1. Run both models on a 50x50 grid
  2. Measure complexity at regular intervals
  3. Plot complexity over time for both models
  4. Observe: interacting shows rise-and-fall, non-interacting stays flat

Paper reference: Sections 3, 5.2, 6.2
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import CoffeeAutomaton, measure_complexity, measure_entropy


def run_and_measure(model: str, grid_size: int = 50,
                    total_steps: int = 50000, num_snapshots: int = 50):
    """
    TODO: Run a coffee automaton simulation and collect measurements.

    Args:
        model: 'interacting' or 'non_interacting'
        grid_size: N
        total_steps: how many steps total
        num_snapshots: how many measurement points

    Returns:
        times: list of time steps
        complexities: list of complexity measurements
        entropies: list of entropy measurements
    """
    # YOUR CODE HERE
    # Hint:
    #   1. Create a CoffeeAutomaton with the given model
    #   2. Calculate steps_per_snapshot = total_steps // num_snapshots
    #   3. Loop: step(), then measure complexity and entropy
    #   4. Return the collected data
    pass


def plot_comparison(times_inter, cx_inter, times_non, cx_non):
    """
    TODO: Plot complexity over time for both models on the same axes.

    The interacting model should show a clear rise-and-fall.
    The non-interacting model should stay relatively flat.
    """
    import matplotlib.pyplot as plt

    # YOUR CODE HERE
    # Hint: plt.plot(times_inter, cx_inter, label='Interacting')
    #       plt.plot(times_non, cx_non, label='Non-interacting')
    pass


if __name__ == '__main__':
    print("Running interacting model...")
    t_i, cx_i, e_i = run_and_measure('interacting')

    print("Running non-interacting model...")
    t_n, cx_n, e_n = run_and_measure('non_interacting')

    # Print summary
    if cx_i is not None and cx_n is not None:
        print(f"\nInteracting: peak complexity = {max(cx_i)}")
        print(f"Non-interacting: peak complexity = {max(cx_n)}")
        print(f"Ratio: {max(cx_i) / max(cx_n):.1f}x")
        print("\nThe interacting model should have significantly higher peak complexity.")

        plot_comparison(t_i, cx_i, t_n, cx_n)
