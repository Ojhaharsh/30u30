"""
Exercise 1: Kolmogorov Complexity Approximation via gzip

Aaronson discusses using gzip compressed size as a crude approximation
to Kolmogorov complexity. In this exercise, you'll build this tool and
verify it on known signals.

Reference: Aaronson (2011, blog post), Lauren Ouellette's experiments

Tasks:
1. Implement gzip_complexity(data: bytes) -> int
2. Test on three known cases:
   a. All-zeros (highly compressible, low KC)
   b. Random bytes (incompressible, high KC)
   c. Repeating pattern (intermediate KC)
3. Plot compressed size vs. regularity
4. Compare gzip levels 1 vs 9

Expected result: random > pattern > uniform, with gzip level 9
giving tighter (lower) bounds than level 1.
"""

import numpy as np
import gzip


def gzip_complexity(data: bytes, level: int = 9) -> int:
    """
    Approximate Kolmogorov complexity using gzip compressed size.

    Args:
        data: Raw bytes to compress.
        level: Compression level (1-9). Use 9 for best approximation.

    Returns:
        Length of compressed data in bytes.

    TODO: Implement this function.
    Hint: gzip.compress(data, compresslevel=level) returns compressed bytes.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement gzip_complexity")


def test_known_signals():
    """
    Test gzip_complexity on three signals with known properties.

    TODO:
    1. Create 1024 bytes of all-zeros
    2. Create 1024 random bytes
    3. Create 1024 bytes of repeating pattern (e.g., b"ABCD" * 256)
    4. Print compressed sizes for each
    5. Verify: random > pattern > zeros
    """
    # YOUR CODE HERE
    raise NotImplementedError("Create and test the three signals")


def compare_compression_levels():
    """
    Compare gzip levels 1 and 9 on the same data.

    TODO:
    1. Create a 4096-byte repeating pattern
    2. Compress at levels 1, 3, 5, 7, 9
    3. Print compressed size at each level
    4. Verify: level 9 gives smallest (tightest upper bound)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Compare compression levels")


if __name__ == "__main__":
    print("Exercise 1: KC Approximation via gzip")
    print("=" * 40)
    print()

    print("Task 1: Testing known signals...")
    test_known_signals()
    print()

    print("Task 2: Comparing compression levels...")
    compare_compression_levels()
