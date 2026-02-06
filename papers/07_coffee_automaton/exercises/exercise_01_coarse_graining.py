"""
Exercise 1: Coarse-Graining and Complexity Measurement

Build the paper's core measurement tool from scratch.

The paper (Section 5.1) measures "apparent complexity" by:
  1. Dividing the grid into g x g blocks
  2. Averaging each block
  3. Thresholding into discrete buckets
  4. Compressing with gzip
  5. The compressed size = complexity estimate

Your task: implement coarse_grain() and measure both entropy and complexity
on a coffee automaton grid at different time steps.

Paper reference: Sections 4 and 5
"""

import sys
import os
import gzip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import CoffeeAutomaton


def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    TODO: Implement coarse-graining.

    Given an N x N binary grid and a grain size g, produce a (N//g) x (N//g)
    array where each cell is the average of the corresponding g x g block
    in the original grid.

    Returns float array with values in [0, 1].
    """
    # YOUR CODE HERE
    pass


def threshold(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """
    TODO: Threshold floating-point values into discrete buckets.

    Map [0, 1] values to integers in [0, num_buckets-1].
    For 3 buckets: 0 = mostly coffee, 1 = mixed, 2 = mostly cream.

    Returns uint8 array.
    """
    # YOUR CODE HERE
    pass


def gzip_complexity(grid: np.ndarray, grain_size: int = 10,
                    num_buckets: int = 3) -> int:
    """
    TODO: Measure apparent complexity.

    1. Coarse-grain the grid
    2. Threshold into buckets
    3. Compress with gzip
    4. Return compressed size in bytes
    """
    # YOUR CODE HERE
    pass


def gzip_entropy(grid: np.ndarray) -> int:
    """
    TODO: Measure entropy estimate.

    Compress the fine-grained grid directly with gzip.
    Return compressed size in bytes.
    """
    # YOUR CODE HERE
    pass


if __name__ == '__main__':
    # Test your implementation:
    # Create a 50x50 automaton and measure at t=0, t=10000, t=50000
    ca = CoffeeAutomaton(grid_size=50, model='interacting', seed=42)

    print("Time | Entropy | Complexity")
    print("-----|---------|----------")

    for t in [0, 10000, 50000]:
        if t > 0:
            ca.step(t - ca.time)
        grid = ca.get_binary_grid()
        e = gzip_entropy(grid)
        c = gzip_complexity(grid, grain_size=5, num_buckets=3)
        print(f"{t:>5} | {e:>7} | {c:>10}")

    # Expected behavior:
    # - Entropy should increase over time
    # - Complexity should be low at t=0, higher at t=10000, lower at t=50000
    print("\nIf complexity peaks at intermediate t, your implementation is working.")
