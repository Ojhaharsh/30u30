"""
Solution 5: Compressor Comparison (Figure 5)
"""

import sys
import os
import gzip
import bz2
import lzma
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array,
    row_majority_adjust, grid_to_bytes
)


def compress_with_all(data: bytes) -> dict:
    """Compress data with multiple algorithms."""
    return {
        'gzip': len(gzip.compress(data, compresslevel=9)),
        'bz2': len(bz2.compress(data, compresslevel=9)),
        'lzma': len(lzma.compress(data)),
        'zlib': len(zlib.compress(data, level=9)),
    }


def run_compressor_comparison(grid_size=50, total_steps=50000, num_snapshots=60):
    """Run the automaton and collect complexity estimates from each compressor."""
    ca = CoffeeAutomaton(grid_size, 'interacting', seed=42)
    grain_size = max(2, grid_size // 10)
    steps_per = max(1, total_steps // num_snapshots)

    times = []
    results = {'gzip': [], 'bz2': [], 'lzma': [], 'zlib': []}

    # Initial measurement
    grid = ca.get_binary_grid()
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, 7)
    discrete = row_majority_adjust(discrete)
    data = grid_to_bytes(discrete)

    times.append(0)
    for name, size in compress_with_all(data).items():
        results[name].append(size)

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)

        grid = ca.get_binary_grid()
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, 7)
        discrete = row_majority_adjust(discrete)
        data = grid_to_bytes(discrete)

        for name, size in compress_with_all(data).items():
            results[name].append(size)

    return times, results


def plot_comparison(times, results):
    """Plot complexity curves for all compressors."""
    import matplotlib.pyplot as plt

    colors = {'gzip': 'blue', 'bz2': 'red', 'lzma': 'green', 'zlib': 'orange'}

    plt.figure(figsize=(10, 6))
    for name, sizes in results.items():
        plt.plot(times, sizes, color=colors.get(name, 'black'),
                 label=name, linewidth=1.5)

    plt.xlabel('Time step')
    plt.ylabel('Compressed size (bytes)')
    plt.title('Complexity with Different Compressors (cf. Figure 5)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Compressor Comparison (Figure 5)")
    print("=" * 50)

    times, compressor_results = run_compressor_comparison()

    print("\nPeak complexity by compressor:")
    for name, sizes in compressor_results.items():
        peak = max(sizes)
        peak_t = times[sizes.index(peak)]
        print(f"  {name:>6}: {peak:>6} bytes at t={peak_t}")

    print("\nAll compressors should show the same qualitative rise-and-fall.")

    plot_comparison(times, compressor_results)
