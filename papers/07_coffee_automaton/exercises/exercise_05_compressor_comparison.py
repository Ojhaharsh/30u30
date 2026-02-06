"""
Exercise 5: Compressor Comparison

The paper (Figure 5) shows that different file compression programs produce
qualitatively similar complexity curves.

"The results achieved using the coarse-graining metric are qualitatively similar
when different compression programs are used, as shown in Figure 5." (Section 5.2)

Your task:
  1. Run the interacting coffee automaton
  2. At each snapshot, compress the coarse-grained state with multiple compressors
  3. Plot complexity curves for each compressor
  4. Verify they all show the rise-and-fall pattern

Available compressors in Python:
  - gzip (LZ77 + Huffman)
  - bz2 (Burrows-Wheeler + Huffman)
  - lzma (LZMA2)
  - zlib (same as gzip, different wrapper)

Paper reference: Section 5.2, Figure 5
"""

import sys
import os
import gzip
import bz2
import lzma
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array,
    row_majority_adjust, grid_to_bytes
)


def compress_with_all(data: bytes) -> dict:
    """
    TODO: Compress data with multiple algorithms.

    Return a dict with compressor name -> compressed size.
    Use: gzip, bz2, lzma, zlib
    """
    # YOUR CODE HERE
    pass


def run_compressor_comparison(grid_size=50, total_steps=50000, num_snapshots=60):
    """
    TODO: Run the automaton and collect complexity estimates from each compressor.

    Returns:
        times: list of time steps
        results: dict of {compressor_name: [sizes at each time step]}
    """
    # YOUR CODE HERE
    # Hint:
    #   1. Create CoffeeAutomaton
    #   2. At each snapshot, get grid, coarse-grain, threshold, adjust
    #   3. Convert to bytes
    #   4. Compress with all compressors
    #   5. Collect sizes
    pass


def plot_comparison(times, results):
    """
    TODO: Plot complexity curves for all compressors.

    All curves should show the same qualitative rise-and-fall pattern,
    even though the absolute values differ.
    """
    import matplotlib.pyplot as plt

    # YOUR CODE HERE
    pass


if __name__ == '__main__':
    print("Compressor Comparison (Figure 5)")
    print("=" * 50)

    result = run_compressor_comparison()

    if result is not None:
        times, compressor_results = result
        print("\nPeak complexity by compressor:")
        for name, sizes in compressor_results.items():
            peak = max(sizes)
            peak_t = times[sizes.index(peak)] if isinstance(sizes, list) else times[np.argmax(sizes)]
            print(f"  {name:>6}: {peak:>6} bytes at t={peak_t}")

        plot_comparison(times, compressor_results)
