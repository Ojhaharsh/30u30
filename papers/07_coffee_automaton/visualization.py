"""
visualization.py — Plotting tools for the coffee automaton experiments.

Generates figures analogous to those in the paper:
  - Entropy and complexity over time (Figures 2, 10)
  - Grid state visualization at key moments (Figures 3, 4, 11, 12)
  - Scaling analysis (Figures 6-8)
  - Compressor comparison (Figure 5)

Paper: Aaronson, Carroll, Ouellette (2014), arXiv:1405.6903
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    CoffeeAutomaton, run_simulation, coarse_grain,
    threshold_array, row_majority_adjust, gzip_size, grid_to_bytes
)


def plot_entropy_complexity(results: Dict, title: str = 'Coffee Automaton',
                            ax: Optional[plt.Axes] = None):
    """
    Plot entropy and complexity on the same time axis.
    Replicates the left/right panels of Figures 2 and 10.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(results['times'], results['entropy'], 'b-', label='Entropy (gzip fine-grained)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Compressed size (bytes)', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(results['times'], results['complexity'], 'r-', label='Complexity (gzip coarse-grained)')
    ax2.set_ylabel('Complexity (bytes)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Mark complexity peak
    peak_idx = np.argmax(results['complexity'])
    ax2.axvline(results['times'][peak_idx], color='gray', linestyle='--', alpha=0.4)

    ax.set_title(title)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

    return ax


def plot_grid_snapshots(results: Dict, num_snapshots: int = 5,
                        grain_size: int = 10, num_buckets: int = 7,
                        title: str = 'Grid State Over Time'):
    """
    Visualize the automaton state at key moments.
    Replicates Figures 3, 4, 11, 12.

    Shows fine-grained (top row) and coarse-grained (bottom row) states.
    """
    if 'grids' not in results or len(results['grids']) == 0:
        print("No grid snapshots stored (grid_size > 100?)")
        return

    grids = results['grids']
    times = results['times']
    complexities = results['complexity']

    # Pick snapshots: start, peak, and evenly spaced
    peak_idx = np.argmax(complexities)
    total = len(grids)
    indices = [0, peak_idx]

    # Add evenly spaced indices
    step = total // (num_snapshots - 1)
    for i in range(1, num_snapshots - 1):
        idx = i * step
        if idx not in indices:
            indices.append(idx)
    indices.append(total - 1)
    indices = sorted(set(indices))[:num_snapshots]

    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))

    for col, idx in enumerate(indices):
        grid = grids[idx]
        t = times[idx]

        # Fine-grained (top row)
        axes[0, col].imshow(grid, cmap='gray_r', vmin=0, vmax=1)
        label = f't={t}'
        if idx == peak_idx:
            label += ' (peak)'
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis('off')

        # Coarse-grained (bottom row)
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, num_buckets)
        discrete = row_majority_adjust(discrete)
        axes[1, col].imshow(discrete, cmap='gray_r', vmin=0, vmax=num_buckets - 1)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Fine-grained', fontsize=10)
    axes[1, 0].set_ylabel('Coarse-grained', fontsize=10)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_model_comparison(interacting: Dict, non_interacting: Dict):
    """
    Side-by-side comparison of interacting vs non-interacting.
    Replicates Figures 2 and 10 layout.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Interacting
    axes[0, 0].plot(interacting['times'], interacting['entropy'], 'b-')
    axes[0, 0].set_title('Interacting: Entropy')
    axes[0, 0].set_ylabel('gzip size (bytes)')

    axes[0, 1].plot(interacting['times'], interacting['complexity'], 'r-')
    peak = np.argmax(interacting['complexity'])
    axes[0, 1].axvline(interacting['times'][peak], color='gray', ls='--', alpha=0.4)
    axes[0, 1].set_title('Interacting: Complexity')
    axes[0, 1].set_ylabel('gzip size (bytes)')

    # Non-interacting
    axes[1, 0].plot(non_interacting['times'], non_interacting['entropy'], 'b-')
    axes[1, 0].set_title('Non-Interacting: Entropy')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('gzip size (bytes)')

    axes[1, 1].plot(non_interacting['times'], non_interacting['complexity'], 'r-')
    axes[1, 1].set_title('Non-Interacting: Complexity')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('gzip size (bytes)')

    plt.suptitle('Coffee Automaton: Interacting vs Non-Interacting\n'
                 '(cf. Figures 2 and 10, Aaronson et al. 2014)')
    plt.tight_layout()
    return fig


def plot_scaling_analysis(scaling_results: Dict):
    """
    Plot scaling behavior — replicates Figures 6-8.
    Max entropy ~ n^2, Max complexity ~ n, Time to peak ~ n^2.
    """
    sizes = scaling_results['sizes']
    x_fit = np.linspace(sizes[0], sizes[-1], 50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Figure 6: Max entropy vs n
    axes[0].plot(sizes, scaling_results['max_entropy'], 'bo-', markersize=6)
    c = np.polyfit(sizes, scaling_results['max_entropy'], 2)
    r2 = 1 - np.sum((scaling_results['max_entropy'] - np.polyval(c, sizes))**2) / \
         np.sum((scaling_results['max_entropy'] - np.mean(scaling_results['max_entropy']))**2)
    axes[0].plot(x_fit, np.polyval(c, x_fit), 'b--', alpha=0.5,
                 label=f'Quadratic fit (r^2={r2:.4f})')
    axes[0].set_xlabel('Grid size n')
    axes[0].set_ylabel('Max entropy (bytes)')
    axes[0].set_title('Max Entropy ~ n^2 (Figure 6)')
    axes[0].legend(fontsize=8)

    # Figure 7: Max complexity vs n
    axes[1].plot(sizes, scaling_results['max_complexity'], 'ro-', markersize=6)
    c = np.polyfit(sizes, scaling_results['max_complexity'], 1)
    r2 = 1 - np.sum((scaling_results['max_complexity'] - np.polyval(c, sizes))**2) / \
         np.sum((scaling_results['max_complexity'] - np.mean(scaling_results['max_complexity']))**2)
    axes[1].plot(x_fit, np.polyval(c, x_fit), 'r--', alpha=0.5,
                 label=f'Linear fit (r^2={r2:.4f})')
    axes[1].set_xlabel('Grid size n')
    axes[1].set_ylabel('Max complexity (bytes)')
    axes[1].set_title('Max Complexity ~ n (Figure 7)')
    axes[1].legend(fontsize=8)

    # Figure 8: Time to peak vs n
    axes[2].plot(sizes, scaling_results['peak_time'], 'go-', markersize=6)
    c = np.polyfit(sizes, scaling_results['peak_time'], 2)
    r2 = 1 - np.sum((scaling_results['peak_time'] - np.polyval(c, sizes))**2) / \
         np.sum((scaling_results['peak_time'] - np.mean(scaling_results['peak_time']))**2)
    axes[2].plot(x_fit, np.polyval(c, x_fit), 'g--', alpha=0.5,
                 label=f'Quadratic fit (r^2={r2:.4f})')
    axes[2].set_xlabel('Grid size n')
    axes[2].set_ylabel('Time to peak')
    axes[2].set_title('Time to Peak ~ n^2 (Figure 8)')
    axes[2].legend(fontsize=8)

    plt.suptitle('Scaling Analysis — Figures 6-8 from Aaronson et al. (2014)')
    plt.tight_layout()
    return fig


def plot_compressor_comparison(grid_size: int = 50, total_steps: int = 100000,
                               num_snapshots: int = 80, seed: int = 42):
    """
    Compare different compression algorithms — replicates Figure 5.

    The paper shows that gzip, bzip2, etc. all produce qualitatively
    similar complexity curves.
    """
    import bz2
    import lzma

    # Run simulation once, collect grids
    ca = CoffeeAutomaton(grid_size, 'interacting', seed)
    grain_size = max(2, grid_size // 10)
    steps_per = max(1, total_steps // num_snapshots)

    times = [0]
    gzip_cx = [0]
    bz2_cx = [0]
    lzma_cx = [0]

    # Initial measurement
    grid = ca.get_binary_grid()
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, 7)
    discrete = row_majority_adjust(discrete)
    data = grid_to_bytes(discrete)

    gzip_cx[0] = len(gzip.compress(data, compresslevel=9))
    bz2_cx[0] = len(bz2.compress(data, compresslevel=9))
    lzma_cx[0] = len(lzma.compress(data))

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)

        grid = ca.get_binary_grid()
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, 7)
        discrete = row_majority_adjust(discrete)
        data = grid_to_bytes(discrete)

        gzip_cx.append(len(gzip.compress(data, compresslevel=9)))
        bz2_cx.append(len(bz2.compress(data, compresslevel=9)))
        lzma_cx.append(len(lzma.compress(data)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, gzip_cx, 'b-', label='gzip')
    ax.plot(times, bz2_cx, 'r-', label='bzip2')
    ax.plot(times, lzma_cx, 'g-', label='lzma')

    ax.set_xlabel('Time step')
    ax.set_ylabel('Compressed size of coarse-grained state (bytes)')
    ax.set_title('Complexity Estimates Using Different Compressors\n'
                 '(cf. Figure 5, Aaronson et al. 2014)')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_complexity_curve_annotated(results: Dict, model: str = 'interacting'):
    """
    A single annotated complexity curve showing the rise and fall.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    times = results['times']
    cx = results['complexity']

    ax.plot(times, cx, 'r-', linewidth=2)
    ax.fill_between(times, cx, alpha=0.1, color='r')

    peak_idx = np.argmax(cx)
    peak_t = times[peak_idx]
    peak_v = cx[peak_idx]

    ax.annotate(f'Peak: {peak_v} bytes\nt = {peak_t}',
                xy=(peak_t, peak_v),
                xytext=(peak_t + (times[-1] - times[0]) * 0.1, peak_v * 0.9),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    # Mark phases
    third = len(times) // 3
    ax.axvspan(times[0], times[third], alpha=0.05, color='blue', label='Separated')
    ax.axvspan(times[third], times[2 * third], alpha=0.05, color='red', label='Complex structures')
    ax.axvspan(times[2 * third], times[-1], alpha=0.05, color='green', label='Well-mixed')

    ax.set_xlabel('Time step')
    ax.set_ylabel('Apparent complexity (bytes)')
    ax.set_title(f'Rise and Fall of Complexity — {model.title()} Model')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main: generate all key figures
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Generating visualizations for the Coffee Automaton paper...")
    print("=" * 55)

    # Run a quick simulation
    print("\n1. Running interacting simulation (50x50, 50000 steps)...")
    results = run_simulation(grid_size=50, total_steps=50000, num_snapshots=80,
                             model='interacting', seed=42)

    print("2. Plotting entropy and complexity...")
    plot_entropy_complexity(results, 'Interacting Model (50x50)')

    print("3. Plotting grid snapshots...")
    plot_grid_snapshots(results, num_snapshots=5, grain_size=5)

    print("4. Plotting annotated complexity curve...")
    plot_complexity_curve_annotated(results)

    print("\n5. Running compressor comparison...")
    plot_compressor_comparison(grid_size=50, total_steps=50000, num_snapshots=60)

    plt.show()
    print("\nDone.")
