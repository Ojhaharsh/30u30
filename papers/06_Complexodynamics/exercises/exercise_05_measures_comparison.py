"""
Exercise 5: Complexity Measures Comparison

The capstone exercise: run ALL complexity measures on the same coffee
mixing simulation and compare their curves. The goal is to see which
measures show the characteristic "hump" most clearly.

Measures to compare:
1. gzip complexity (raw KC proxy)
2. Coarse-grained KC at multiple scales (Carroll's idea)
3. Two-part code Part 1 (sophistication proxy)
4. Boundary fraction (interface length)
5. Shannon entropy (should NOT show the hump -- it's monotone)

Reference: Aaronson (2011, blog post), discussion of which measure is "right"

Tasks:
1. Run a single simulation
2. Measure all 5 quantities at each time step
3. Plot all on a single normalized figure
4. Identify which measure shows the clearest hump
5. Discuss: which is closest to Aaronson's complextropy?
"""

import numpy as np
import gzip
import sys
import os

# Add parent directory to path for importing implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_comparison(grid_size=64, n_steps=30000, swaps_per_step=50,
                   measure_interval=300):
    """
    Run simulation and collect all complexity measures.

    Returns:
        Dict with keys: 'times', 'gzip', 'coarse_kc_4', 'coarse_kc_8',
        'part1', 'boundary', 'entropy'

    TODO:
    1. Import or implement: create_grid, batch_swap, gzip_complexity,
       coarse_grained_kc, two_part_code, grid_entropy, boundary_fraction
    2. Run simulation, measure at intervals
    3. Return all data as a dict
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement run_comparison")


def normalize_to_01(values):
    """Normalize a list of values to [0, 1] range."""
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def plot_comparison(data):
    """
    Plot all measures on a single normalized figure.

    TODO:
    1. Normalize each measure to [0, 1]
    2. Plot all on same axes
    3. Mark the peak time for each
    4. Add legend
    5. Note which measures show the hump and which don't
    """
    # YOUR CODE HERE
    raise NotImplementedError("Plot comparison")


def analyze_results(data):
    """
    Print analysis of which measures show the hump.

    For each measure:
    - Peak time
    - Peak/initial ratio
    - Peak/final ratio
    - Whether it's monotone (no hump) or has a clear peak

    TODO: Implement this analysis.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Analyze results")


if __name__ == "__main__":
    print("Exercise 5: Complexity Measures Comparison")
    print("=" * 40)
    print()

    print("Running simulation with all measures...")
    data = run_comparison()
    print()

    print("Analysis:")
    analyze_results(data)
    print()

    print("Plotting comparison...")
    plot_comparison(data)
    print()

    print("Questions to consider:")
    print("1. Which measure shows the sharpest hump?")
    print("2. Do gzip and sophistication peak at the same time?")
    print("3. Is boundary fraction a good complexity proxy? Why or why not?")
    print("4. Which measure is closest to Aaronson's complextropy?")
