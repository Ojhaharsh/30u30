"""
Solution 5: Complexity Measures Comparison
"""

import numpy as np
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from implementation import (
    create_initial_grid, batch_swap, gzip_complexity,
    coarse_grained_kc, two_part_code, grid_entropy, boundary_fraction,
)


def normalize_to_01(values):
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def run_comparison(grid_size=64, n_steps=30000, swaps_per_step=50,
                   measure_interval=300):
    """Run simulation and collect all complexity measures."""
    np.random.seed(42)
    grid = create_initial_grid(grid_size)

    data = {
        'times': [],
        'gzip': [],
        'coarse_kc_4': [],
        'coarse_kc_8': [],
        'part1': [],
        'boundary': [],
        'entropy': [],
    }

    def measure(step):
        data['times'].append(step * swaps_per_step)
        data['gzip'].append(gzip_complexity(grid))
        data['coarse_kc_4'].append(coarse_grained_kc(grid, 4))
        data['coarse_kc_8'].append(coarse_grained_kc(grid, 8))
        p1, _ = two_part_code(grid, block_size=4)
        data['part1'].append(p1)
        data['boundary'].append(boundary_fraction(grid))
        data['entropy'].append(grid_entropy(grid))

    measure(0)
    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            measure(step)
            if step % (measure_interval * 10) == 0:
                print(f"  Step {step}/{n_steps}")

    return data


def analyze_results(data):
    """Print analysis of each measure."""
    measures = {
        'gzip (KC proxy)': data['gzip'],
        'coarse KC (bs=4)': data['coarse_kc_4'],
        'coarse KC (bs=8)': data['coarse_kc_8'],
        'sophistication (Part 1)': data['part1'],
        'boundary fraction': data['boundary'],
        'entropy': data['entropy'],
    }

    times = data['times']

    for name, values in measures.items():
        arr = np.array(values)
        peak_idx = np.argmax(arr)
        is_monotone = all(arr[i] <= arr[i+1] + 0.01 for i in range(len(arr)-1))

        initial = arr[0]
        peak = arr[peak_idx]
        final = arr[-1]

        print(f"  {name}:")
        print(f"    Initial={initial:.1f}, Peak={peak:.1f} (t={times[peak_idx]:,}), "
              f"Final={final:.1f}")
        if is_monotone:
            print(f"    --> MONOTONE (no hump)")
        elif peak > initial and peak > final:
            ratio = peak / max(initial, 0.001)
            print(f"    --> HUMP detected (peak/initial = {ratio:.2f}x)")
        else:
            print(f"    --> Unclear pattern")
        print()


def plot_comparison(data):
    """Plot all measures normalized on single figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available)")
        return

    times = data['times']

    measures = {
        'gzip (KC)': ('r', data['gzip']),
        'coarse KC (bs=4)': ('orange', data['coarse_kc_4']),
        'sophistication': ('g', data['part1']),
        'boundary': ('b', data['boundary']),
        'entropy': ('purple', data['entropy']),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw values
    for name, (color, values) in measures.items():
        axes[0].plot(times, values, color=color, linewidth=1.5, label=name)
    axes[0].set_title("Raw Values")
    axes[0].set_xlabel("Total swaps")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Normalized
    for name, (color, values) in measures.items():
        normed = normalize_to_01(values)
        axes[1].plot(times, normed, color=color, linewidth=1.5, label=name)
    axes[1].set_title("Normalized [0, 1]")
    axes[1].set_xlabel("Total swaps")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "All Complexity Measures Comparison\n"
        "Entropy is monotone; others should show the 'hump'",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig("measures_comparison.png", dpi=150)
    print("Saved: measures_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Solution 5: Complexity Measures Comparison")
    print("=" * 40)
    print()

    print("Running simulation with all measures...")
    data = run_comparison()
    print()

    print("Analysis:")
    analyze_results(data)

    print("Plotting comparison...")
    plot_comparison(data)
    print()

    print("Conclusions:")
    print("- Entropy is monotone (no hump) -- as expected")
    print("- gzip, coarse KC, and sophistication proxy all show the hump")
    print("- Sophistication (Part 1) is closest to Aaronson's complextropy")
    print("- Boundary fraction also peaks, but is a geometric measure,")
    print("  not an information-theoretic one")
