"""
Solution 3: Multi-Scale Complexity Analysis
"""

import numpy as np
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from implementation import create_initial_grid, batch_swap


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def coarse_grain(grid: np.ndarray, block_size: int) -> np.ndarray:
    """Average grid values over block_size x block_size blocks."""
    h, w = grid.shape
    h_new = h // block_size
    w_new = w // block_size
    result = np.zeros((h_new, w_new))
    for i in range(h_new):
        for j in range(w_new):
            block = grid[i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size]
            result[i, j] = block.mean()
    return result


def coarse_grained_kc(grid: np.ndarray, block_size: int) -> int:
    """KC of coarse-grained grid."""
    coarse = coarse_grain(grid, block_size)
    quantized = (coarse * 255).astype(np.uint8)
    return gzip_size(quantized.tobytes())


def multiscale_analysis(grid: np.ndarray) -> dict:
    """Compute KC at multiple scales."""
    grid_size = min(grid.shape)
    scales = [2**k for k in range(1, int(np.log2(grid_size)))]
    result = {}
    for s in scales:
        if grid_size // s >= 2:
            result[s] = coarse_grained_kc(grid, s)
    return result


def run_multiscale_experiment(grid_size=64, n_steps=20000, swaps_per_step=50):
    """Run coffee simulation and track multi-scale KC."""
    np.random.seed(42)
    grid = create_initial_grid(grid_size)

    max_power = int(np.log2(grid_size)) - 1
    scales = [2**k for k in range(1, max_power + 1)]
    scale_data = {s: [] for s in scales}
    times = []

    measure_interval = max(1, n_steps // 50)

    # Initial measurement
    times.append(0)
    ms = multiscale_analysis(grid)
    for s in scales:
        scale_data[s].append(ms.get(s, 0))

    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            times.append(step * swaps_per_step)
            ms = multiscale_analysis(grid)
            for s in scales:
                scale_data[s].append(ms.get(s, 0))

    return times, scale_data


if __name__ == "__main__":
    print("Solution 3: Multi-Scale Complexity Analysis")
    print("=" * 40)
    print()

    print("Running experiment...")
    times, scale_data = run_multiscale_experiment()
    print(f"  Time points: {len(times)}")
    print(f"  Scales: {sorted(scale_data.keys())}")

    for scale, values in sorted(scale_data.items()):
        peak_idx = np.argmax(values)
        print(f"  Scale {scale}: peak at t={times[peak_idx]:,}, "
              f"val={values[peak_idx]}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for scale, values in sorted(scale_data.items()):
            axes[0].plot(times, values, label=f"scale={scale}")
        axes[0].set_xlabel("Total swaps")
        axes[0].set_ylabel("Compressed bytes")
        axes[0].set_title("KC at Different Scales")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        scales_sorted = sorted(scale_data.keys())
        matrix = np.array([scale_data[s] for s in scales_sorted])
        im = axes[1].imshow(matrix, aspect='auto', cmap='hot',
                           interpolation='bilinear')
        axes[1].set_yticks(range(len(scales_sorted)))
        axes[1].set_yticklabels([str(s) for s in scales_sorted])
        axes[1].set_title("Multi-Scale Heatmap (Trevisan)")
        axes[1].set_xlabel("Time index")
        axes[1].set_ylabel("Scale")
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig("multiscale_analysis.png", dpi=100)
        print("Saved: multiscale_analysis.png")
    except ImportError:
        print("(matplotlib not available)")
