"""
Exercise 4: Two-Part Codes (Sophistication Proxy)

Sophistication (Koppel 1988, Gacs-Tromp-Vitanyi) measures the complexity
of the "model" part of a two-part description. Aaronson's complextropy is
a resource-bounded version of this.

We approximate sophistication using a two-part code:
  Part 1: Coarse-grained description (the "model")
  Part 2: Residual (what's left to reconstruct the exact state)

The size of Part 1 should show the characteristic hump: low at t=0 (trivial
model), high at intermediate times (complex boundaries), low at equilibrium.

Reference: Aaronson (2011, blog post), sophistication section

Tasks:
1. Implement two_part_code(grid, block_size) -> (part1_size, part2_size)
2. Verify Part 1 is small for simple grids (all-black, all-white)
3. Verify Part 1 is small for random grids (model = "everything")
4. Show Part 1 peaks for structured grids (intermediate mixing)
5. Compare Part 1 trajectory to raw gzip trajectory
"""

import numpy as np
import gzip


def gzip_size(data: bytes) -> int:
    """Compressed size via gzip."""
    return len(gzip.compress(data, compresslevel=9))


def two_part_code(grid: np.ndarray, block_size: int = 4):
    """
    Compute two-part code for a 2D grid.

    Part 1 (model): coarse-grained version, compressed.
    Part 2 (residual): difference between reconstruction and actual, compressed.

    Args:
        grid: 2D binary (0/1) array.
        block_size: Coarse-graining scale.

    Returns:
        (part1_size, part2_size) in bytes.

    TODO: Implement this function.
    Steps:
    1. Coarse-grain the grid by averaging block_size x block_size blocks
    2. Quantize coarse grid to uint8, compress = Part 1
    3. Reconstruct from coarse (expand back to original size)
    4. Compute residual = original - reconstruction
    5. Map residual from [-1,1] to [0,255], compress = Part 2
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement two_part_code")


def verify_on_simple_grids():
    """
    Verify that Part 1 (sophistication proxy) is small for both
    simple and random grids, but large for structured grids.

    TODO:
    1. Create: all-zeros grid, all-ones grid, random grid, checkerboard
    2. Compute two_part_code for each
    3. Print Part 1 sizes
    4. Verify: simple and random have small Part 1
    """
    # YOUR CODE HERE
    raise NotImplementedError("Verify on simple grids")


def sophistication_trajectory(grid_size=64, n_steps=20000, swaps_per_step=50):
    """
    Track Part 1 size (sophistication proxy) over time during coffee mixing.

    Returns:
        (times, part1_values, gzip_values) for plotting.

    TODO: Run coffee simulation, measure Part 1 and gzip at intervals,
    show that Part 1 also has a hump.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement trajectory")


if __name__ == "__main__":
    print("Exercise 4: Two-Part Codes (Sophistication Proxy)")
    print("=" * 40)
    print()

    print("Verifying on simple grids...")
    verify_on_simple_grids()
    print()

    print("Computing sophistication trajectory...")
    times, part1_vals, gzip_vals = sophistication_trajectory()

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(times, gzip_vals, 'r-', label='gzip (KC)')
        axes[0].set_title("gzip Complexity")
        axes[0].legend()
        axes[1].plot(times, part1_vals, 'g-', label='Part 1 (sophistication)')
        axes[1].set_title("Sophistication Proxy")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig("sophistication_trajectory.png", dpi=100)
        print("Saved: sophistication_trajectory.png")
    except ImportError:
        print("(matplotlib not available)")
