"""
Exercise 3: Multi-Scale Complexity Analysis

Luca Trevisan (blog comment #17) suggested: divide the grid into blocks
at each scale, compute average content per block, then track KC at each
scale over time. This gives a 2D picture (scale x time) of where the
complexity "bump" lives.

Sean Carroll (#6-7) also emphasized coarse-graining: measure KC of a
blurred version of the state to capture macroscopic structure.

Reference: Aaronson (2011, blog post), comments #6-7 and #17

Tasks:
1. Implement coarse_grain(grid, block_size) -> smaller array
2. Implement coarse_grained_kc(grid, block_size) -> int
3. Run on coffee simulation at multiple scales
4. Plot KC at each scale over time
5. Create the scale-time heatmap Trevisan described
"""

import numpy as np
import gzip


def coarse_grain(grid: np.ndarray, block_size: int) -> np.ndarray:
    """
    Average grid values over block_size x block_size blocks.

    Args:
        grid: 2D array.
        block_size: Side length of blocks.

    Returns:
        Smaller 2D array where each cell is the mean of a block.

    TODO: Implement this function.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement coarse_grain")


def coarse_grained_kc(grid: np.ndarray, block_size: int) -> int:
    """
    KC of the coarse-grained grid.

    Steps:
    1. Coarse-grain the grid
    2. Quantize to uint8 (0-255)
    3. Compress with gzip

    Args:
        grid: 2D binary array.
        block_size: Coarse-graining scale.

    Returns:
        Compressed size in bytes.

    TODO: Implement this function.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement coarse_grained_kc")


def multiscale_analysis(grid: np.ndarray) -> dict:
    """
    Compute KC at multiple coarse-graining scales.

    Args:
        grid: 2D binary array.

    Returns:
        Dict mapping block_size -> compressed size.

    TODO:
    1. Compute scales as powers of 2 up to grid_size/4
    2. For each scale, compute coarse_grained_kc
    3. Return dict
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement multiscale_analysis")


def run_multiscale_experiment(grid_size=64, n_steps=20000, swaps_per_step=50):
    """
    Run coffee simulation and track multi-scale KC over time.

    Returns:
        (times, scale_data) where scale_data[scale] is a list of KC values.

    TODO: Import coffee mixing from exercise 2 or implementation.py,
    run simulation, measure multi-scale KC at intervals.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement experiment")


if __name__ == "__main__":
    print("Exercise 3: Multi-Scale Complexity Analysis")
    print("=" * 40)
    print()

    times, scale_data = run_multiscale_experiment()

    try:
        import matplotlib.pyplot as plt

        # Line plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for scale, values in scale_data.items():
            axes[0].plot(times, values, label=f"scale={scale}")
        axes[0].set_xlabel("Total swaps")
        axes[0].set_ylabel("Compressed bytes")
        axes[0].set_title("KC at Different Scales")
        axes[0].legend()

        # Heatmap
        scales = sorted(scale_data.keys())
        matrix = np.array([scale_data[s] for s in scales])
        axes[1].imshow(matrix, aspect='auto', cmap='hot')
        axes[1].set_yticks(range(len(scales)))
        axes[1].set_yticklabels([str(s) for s in scales])
        axes[1].set_title("Multi-Scale Heatmap (Trevisan)")

        plt.tight_layout()
        plt.savefig("multiscale_analysis.png", dpi=100)
        print("Saved: multiscale_analysis.png")
    except ImportError:
        print("(matplotlib not available, skipping plot)")
