"""
Solution 4: Adjusted Coarse-Graining (Section 6)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array, gzip_size, grid_to_bytes
)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """
    Row-majority adjustment from Section 6.1.
    Snap cells within 1 bucket of the row majority value.
    """
    adjusted = thresholded.copy()
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        values, counts = np.unique(row, return_counts=True)
        majority = values[np.argmax(counts)]
        mask = np.abs(row.astype(np.int16) - int(majority)) <= 1
        adjusted[i, mask] = majority
    return adjusted


def measure_with_method(grid, grain_size, num_buckets, use_adjustment):
    """Measure complexity with specified coarse-graining method."""
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, num_buckets)
    if use_adjustment:
        discrete = row_majority_adjust(discrete)
    return gzip_size(grid_to_bytes(discrete))


if __name__ == '__main__':
    print("Adjusted Coarse-Graining Experiment (Section 6)")
    print("=" * 50)

    grid_size = 50
    total_steps = 50000
    num_snapshots = 50

    for model in ['interacting', 'non_interacting']:
        print(f"\n{model.replace('_', '-').title()} model:")

        ca = CoffeeAutomaton(grid_size, model, seed=42)
        steps_per = total_steps // num_snapshots

        cx_3bucket = []
        cx_7adjusted = []

        for i in range(num_snapshots):
            ca.step(steps_per)
            grid = ca.get_binary_grid()

            c3 = measure_with_method(grid, grain_size=5, num_buckets=3, use_adjustment=False)
            c7 = measure_with_method(grid, grain_size=5, num_buckets=7, use_adjustment=True)

            cx_3bucket.append(c3)
            cx_7adjusted.append(c7)

        print(f"  3-bucket peak: {max(cx_3bucket)}")
        print(f"  7-bucket adjusted peak: {max(cx_7adjusted)}")

    print("\nNon-interacting 7-bucket adjusted should be much lower — artifacts removed.")
