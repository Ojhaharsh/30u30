"""
Solution 4: Two-Part Codes (Sophistication Proxy)
"""

import numpy as np
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from implementation import create_initial_grid, batch_swap, gzip_complexity


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def two_part_code(grid: np.ndarray, block_size: int = 4):
    """Compute two-part code sizes."""
    h, w = grid.shape
    grid_f = grid.astype(np.float64)

    # Coarse-grain
    h_new = h // block_size
    w_new = w // block_size
    coarse = np.zeros((h_new, w_new))
    for i in range(h_new):
        for j in range(w_new):
            block = grid_f[i * block_size:(i + 1) * block_size,
                          j * block_size:(j + 1) * block_size]
            coarse[i, j] = block.mean()

    # Part 1: model
    model_q = (coarse * 255).astype(np.uint8)
    part1 = gzip_size(model_q.tobytes())

    # Reconstruct
    reconstructed = np.repeat(
        np.repeat(coarse, block_size, axis=0),
        block_size, axis=1
    )[:h, :w]

    # Part 2: residual
    residual = grid_f - reconstructed
    residual_uint8 = ((residual + 1.0) * 127.5).astype(np.uint8)
    part2 = gzip_size(residual_uint8.tobytes())

    return part1, part2


def verify_on_simple_grids():
    """Verify sophistication proxy on known grid types."""
    size = 64

    grids = {
        'all-zeros': np.zeros((size, size), dtype=np.uint8),
        'all-ones': np.ones((size, size), dtype=np.uint8),
        'random': np.random.randint(0, 2, (size, size), dtype=np.uint8),
        'checkerboard': np.indices((size, size)).sum(axis=0) % 2,
    }

    for name, grid in grids.items():
        p1, p2 = two_part_code(grid.astype(np.uint8))
        print(f"  {name:15s}: Part1={p1:4d}, Part2={p2:4d}, Total={p1+p2:4d}")

    # Verify: simple grids have small Part 1
    p1_zeros, _ = two_part_code(grids['all-zeros'])
    p1_random, _ = two_part_code(grids['random'])
    print()
    print(f"  [OK] All-zeros Part1 ({p1_zeros}) and random Part1 ({p1_random})")
    print("       Both should be relatively small (simple model or trivial model)")


def sophistication_trajectory(grid_size=64, n_steps=20000, swaps_per_step=50):
    """Track Part 1 size over time during coffee mixing."""
    np.random.seed(42)
    grid = create_initial_grid(grid_size)

    times = [0]
    part1_vals = []
    gzip_vals = []

    p1, _ = two_part_code(grid)
    part1_vals.append(p1)
    gzip_vals.append(gzip_complexity(grid))

    measure_interval = max(1, n_steps // 50)

    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            times.append(step * swaps_per_step)
            p1, _ = two_part_code(grid)
            part1_vals.append(p1)
            gzip_vals.append(gzip_complexity(grid))

    return times, part1_vals, gzip_vals


if __name__ == "__main__":
    print("Solution 4: Two-Part Codes (Sophistication Proxy)")
    print("=" * 40)
    print()

    print("Verifying on simple grids...")
    verify_on_simple_grids()
    print()

    print("Computing sophistication trajectory...")
    times, part1_vals, gzip_vals = sophistication_trajectory()

    peak_soph = np.argmax(part1_vals)
    peak_gzip = np.argmax(gzip_vals)
    print(f"  Sophistication peaks at t={times[peak_soph]:,}")
    print(f"  gzip KC peaks at t={times[peak_gzip]:,}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(times, gzip_vals, 'r-', label='gzip (KC)')
        axes[0].set_title("gzip Complexity")
        axes[0].set_xlabel("Total swaps")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, part1_vals, 'g-', label='Part 1 (sophistication)')
        axes[1].set_title("Sophistication Proxy")
        axes[1].set_xlabel("Total swaps")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("sophistication_trajectory.png", dpi=100)
        print("Saved: sophistication_trajectory.png")
    except ImportError:
        print("(matplotlib not available)")
