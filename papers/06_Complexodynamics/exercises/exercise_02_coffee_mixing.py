"""
Exercise 2: Coffee Mixing Simulation

Build the 2D pixel grid simulation that Aaronson describes in his blog post:
- Start with top half black (coffee), bottom half white (milk)
- At each step, pick a random adjacent coffee-milk pair and swap them
- Track the state over time

This is the model system for testing complexity measures.

Reference: Aaronson (2011, blog post), model system section

Tasks:
1. Implement create_grid(size) -> 2D array
2. Implement swap_step(grid) -> perform one random swap
3. Implement batch_swap(grid, n) -> perform n swaps efficiently
4. Run simulation and save snapshots
5. Verify: system starts separated, develops tendrils, becomes uniform
"""

import numpy as np


def create_grid(size: int = 64) -> np.ndarray:
    """
    Create initial coffee-milk grid.
    Top half = 1 (coffee/black), bottom half = 0 (milk/white).

    Args:
        size: Side length of square grid.

    Returns:
        2D numpy array of 0s and 1s.

    TODO: Implement this function.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_grid")


def swap_step(grid: np.ndarray) -> np.ndarray:
    """
    Perform one random swap: find all adjacent pairs where one is coffee
    and one is milk, pick one uniformly at random, swap them.

    Args:
        grid: 2D binary array (modified in place).

    Returns:
        The grid (same object).

    TODO: Implement this function.
    Hint: Check horizontal and vertical neighbors. Build list of
    swappable pairs, pick one at random, swap.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement swap_step")


def batch_swap(grid: np.ndarray, n_swaps: int = 100) -> np.ndarray:
    """
    Perform n_swaps random swaps efficiently.

    Instead of finding ALL swappable pairs each time (slow), pick a
    random cell and random direction, check if a swap is possible.

    Args:
        grid: 2D binary array (modified in place).
        n_swaps: Number of swaps to attempt.

    Returns:
        The grid (same object).

    TODO: Implement this function.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement batch_swap")


def run_and_snapshot(grid_size=32, n_steps=5000, swaps_per_step=20,
                     snapshot_interval=500):
    """
    Run simulation and collect grid snapshots at intervals.

    Returns:
        List of (step, grid_copy) tuples.

    TODO: Implement this function.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement run_and_snapshot")


if __name__ == "__main__":
    print("Exercise 2: Coffee Mixing Simulation")
    print("=" * 40)
    print()

    snapshots = run_and_snapshot()
    print(f"Collected {len(snapshots)} snapshots")

    # If matplotlib is available, show the progression
    try:
        import matplotlib.pyplot as plt
        n = min(6, len(snapshots))
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)
        for ax, idx in zip(axes, indices):
            step, grid = snapshots[idx]
            ax.imshow(grid, cmap='gray', interpolation='nearest')
            ax.set_title(f"step={step}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("coffee_mixing_snapshots.png", dpi=100)
        print("Saved: coffee_mixing_snapshots.png")
    except ImportError:
        print("(matplotlib not available, skipping plot)")
