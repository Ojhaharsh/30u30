"""
Solution 1: Coarse-Graining and Complexity Measurement
"""

import sys
import os
import gzip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import CoffeeAutomaton


def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    Coarse-grain by averaging over grain_size x grain_size blocks.
    Section 5.1 of the paper.
    """
    n = grid.shape[0]
    cn = n // grain_size
    if cn == 0:
        cn = 1

    coarse = np.zeros((cn, cn), dtype=np.float64)
    for i in range(cn):
        for j in range(cn):
            r0 = i * grain_size
            c0 = j * grain_size
            r1 = min(r0 + grain_size, n)
            c1 = min(c0 + grain_size, n)
            coarse[i, j] = grid[r0:r1, c0:c1].mean()
    return coarse


def threshold(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """
    Threshold into discrete buckets.
    Section 5.1: 3 buckets (mostly coffee, mixed, mostly cream).
    """
    scaled = coarse * (num_buckets - 1)
    return np.clip(np.round(scaled), 0, num_buckets - 1).astype(np.uint8)


def gzip_complexity(grid: np.ndarray, grain_size: int = 10,
                    num_buckets: int = 3) -> int:
    """Apparent complexity: gzip(coarse-grained state)."""
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold(coarse, num_buckets)
    return len(gzip.compress(discrete.tobytes(), compresslevel=9))


def gzip_entropy(grid: np.ndarray) -> int:
    """Entropy estimate: gzip(fine-grained state)."""
    return len(gzip.compress(grid.astype(np.uint8).tobytes(), compresslevel=9))


if __name__ == '__main__':
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

    print("\nComplexity should peak at intermediate time.")
