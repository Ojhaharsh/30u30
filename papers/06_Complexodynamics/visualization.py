"""
visualization.py - Visualization Suite for Coffee Mixing Complexity

Generates key plots demonstrating the "First Law of Complexodynamics":
entropy increases monotonically, but complexity peaks at intermediate times.

Plots generated:
1. The Complexity Hump (gzip size over time)
2. Entropy vs Complexity side-by-side
3. Grid snapshots at key time points
4. Measures comparison (gzip, coarse-grained, two-part code)
5. Multi-scale heatmap
6. Sophistication (Part 1) trajectory
7. Mixing animation frames

Reference: Scott Aaronson (2011) - https://scottaaronson.blog/?p=762

Usage:
    python visualization.py
    python visualization.py --grid-size 64 --steps 30000
"""

import argparse
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
except ImportError:
    print("matplotlib is required for visualization.")
    print("Install with: pip install matplotlib")
    sys.exit(1)

from implementation import (
    run_simulation,
    create_initial_grid,
    batch_swap,
    gzip_complexity,
    coarse_grained_kc,
    two_part_code,
    grid_entropy,
    mean_local_entropy,
    boundary_fraction,
    multiscale_kc,
    sophistication_proxy,
    compute_normalization_bounds,
    normalize_complexity,
)


def plot_complexity_hump(results, output_dir='.'):
    """
    Plot 1: The core result -- complexity (gzip size) over time showing
    the characteristic hump that Aaronson's First Law predicts.
    """
    times = results['times']
    gzip_c = results['gzip_complexity']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, gzip_c, 'r-', linewidth=2, label='gzip complexity')

    # Mark the peak
    peak_idx = np.argmax(gzip_c)
    ax.plot(times[peak_idx], gzip_c[peak_idx], 'ko', markersize=10, zorder=5)
    ax.annotate(
        f'Peak: {gzip_c[peak_idx]} bytes\nat swap {times[peak_idx]:,}',
        xy=(times[peak_idx], gzip_c[peak_idx]),
        xytext=(times[peak_idx] * 1.1, gzip_c[peak_idx] * 0.95),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black'),
    )

    # Annotate regions
    ax.axvspan(0, times[peak_idx] * 0.3, alpha=0.1, color='blue',
               label='Simple (separated)')
    ax.axvspan(times[peak_idx] * 0.6, times[peak_idx] * 1.4, alpha=0.1,
               color='red', label='Complex (tendrils)')
    ax.axvspan(times[-1] * 0.7, times[-1], alpha=0.1, color='green',
               label='Simple (mixed)')

    ax.set_xlabel('Total swaps', fontsize=12)
    ax.set_ylabel('Compressed size (bytes)', fontsize=12)
    ax.set_title(
        'The Complexity Hump\n'
        'Aaronson\'s "First Law of Complexodynamics" (2011)',
        fontsize=14
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{output_dir}/plot1_complexity_hump.png"
    plt.savefig(outpath, dpi=150)
    print(f"  Saved: {outpath}")
    plt.close()


def plot_entropy_vs_complexity(results, output_dir='.'):
    """
    Plot 2: Entropy (monotone) vs Complexity (hump) side by side.
    This is the central contrast of the blog post.
    """
    times = results['times']
    entropy = results['entropy']
    gzip_c = results['gzip_complexity']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entropy
    axes[0].plot(times, entropy, 'b-', linewidth=2)
    axes[0].set_title("Entropy (monotonically increases)", fontsize=13)
    axes[0].set_xlabel("Total swaps")
    axes[0].set_ylabel("Shannon entropy (bits)")
    axes[0].grid(True, alpha=0.3)

    # Complexity
    axes[1].plot(times, gzip_c, 'r-', linewidth=2)
    peak_idx = np.argmax(gzip_c)
    axes[1].axvline(x=times[peak_idx], color='gray', linestyle='--', alpha=0.5)
    axes[1].plot(times[peak_idx], gzip_c[peak_idx], 'ko', markersize=8)
    axes[1].set_title("Complexity (peaks, then decreases)", fontsize=13)
    axes[1].set_xlabel("Total swaps")
    axes[1].set_ylabel("gzip compressed size (bytes)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Entropy vs Complexity: The Central Question\n"
        "Why does 'interestingness' peak while entropy keeps rising?",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot2_entropy_vs_complexity.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_grid_snapshots(results, output_dir='.'):
    """
    Plot 3: Grid snapshots at key time points -- the visual demonstration
    of separated -> tendrils -> mixed.
    """
    grids = results['grids']
    times = results['times']
    n = len(grids)

    # Pick 6 evenly spaced snapshots
    indices = [0] + [int(n * f) for f in [0.1, 0.25, 0.5, 0.75]] + [n - 1]
    indices = [min(i, n - 1) for i in indices]

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3.5))

    for ax, idx in zip(axes, indices):
        ax.imshow(grids[idx], cmap='gray', interpolation='nearest')
        ax.set_title(f"t={times[idx]:,}", fontsize=10)
        ax.axis('off')

    fig.suptitle(
        "Coffee Mixing Over Time\n"
        "Separated (simple) --> Tendrils (complex) --> Mixed (simple)",
        fontsize=13
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot3_grid_snapshots.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_measures_comparison(results, output_dir='.'):
    """
    Plot 4: Compare all complexity measures on the same simulation.
    gzip, coarse-grained KC, sophistication proxy (two-part Part 1).
    """
    times = results['times']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # gzip complexity
    axes[0, 0].plot(times, results['gzip_complexity'], 'r-', linewidth=1.5)
    axes[0, 0].set_title("gzip Complexity (raw KC proxy)")
    axes[0, 0].set_ylabel("Compressed bytes")
    axes[0, 0].grid(True, alpha=0.3)

    # Coarse-grained KC
    for bs, vals in results['coarse_kc'].items():
        if len(vals) > 0 and max(vals) > 0:
            axes[0, 1].plot(times, vals, linewidth=1.5, label=f"block={bs}")
    axes[0, 1].set_title("Coarse-grained KC (Carroll, comment #6)")
    axes[0, 1].set_ylabel("Compressed bytes")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Two-part code
    part1 = [p[0] for p in results['two_part']]
    part2 = [p[1] for p in results['two_part']]
    axes[1, 0].plot(times, part1, 'g-', linewidth=1.5,
                    label='Part 1 (model = sophistication)')
    axes[1, 0].plot(times, part2, 'm--', linewidth=1.5,
                    label='Part 2 (residual = randomness)')
    axes[1, 0].set_title("Two-Part Code (sophistication proxy)")
    axes[1, 0].set_xlabel("Total swaps")
    axes[1, 0].set_ylabel("Compressed bytes")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Boundary fraction
    axes[1, 1].plot(times, results['fraction_mixed'], 'b-', linewidth=1.5)
    axes[1, 1].set_title("Boundary Fraction (interface length)")
    axes[1, 1].set_xlabel("Total swaps")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Comparison of Complexity Measures\n"
        "All should show the 'hump' at intermediate times",
        fontsize=14
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot4_measures_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_multiscale_heatmap(results, output_dir='.'):
    """
    Plot 5: Multi-scale complexity heatmap (Trevisan's idea, comment #17).
    Shows where in scale-space the complexity bump lives.
    """
    times = results['times']
    grid_size = results['grid_size']

    # Recompute multi-scale KC for stored grids
    max_power = int(np.log2(grid_size)) - 1
    scales = [2**k for k in range(1, max_power + 1)]

    matrix = np.zeros((len(scales), len(results['grids'])))
    for t_idx, grid in enumerate(results['grids']):
        ms = multiscale_kc(grid, scales)
        for s_idx, s in enumerate(scales):
            matrix[s_idx, t_idx] = ms.get(s, 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        matrix, aspect='auto', cmap='hot', interpolation='bilinear',
        extent=[times[0], times[-1], len(scales) - 0.5, -0.5]
    )
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([str(s) for s in scales])
    ax.set_xlabel("Total swaps", fontsize=12)
    ax.set_ylabel("Coarse-graining scale", fontsize=12)
    ax.set_title(
        "Multi-Scale Complexity Heatmap (Trevisan, comment #17)\n"
        "Bright = high complexity. The 'bump' at intermediate scales & times.",
        fontsize=13
    )
    plt.colorbar(im, label="Compressed bytes")

    plt.tight_layout()
    outpath = f"{output_dir}/plot5_multiscale_heatmap.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_sophistication_trajectory(results, output_dir='.'):
    """
    Plot 6: Sophistication (Part 1 of two-part code) trajectory.
    This is the closest practical proxy to Aaronson's complextropy.
    """
    times = results['times']
    part1 = [p[0] for p in results['two_part']]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, part1, 'g-', linewidth=2)
    peak_idx = np.argmax(part1)
    ax.plot(times[peak_idx], part1[peak_idx], 'ko', markersize=10)
    ax.annotate(
        f'Peak sophistication\nat swap {times[peak_idx]:,}',
        xy=(times[peak_idx], part1[peak_idx]),
        xytext=(times[peak_idx] * 1.1, part1[peak_idx] * 0.95),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black'),
    )

    ax.set_xlabel("Total swaps", fontsize=12)
    ax.set_ylabel("Part 1 size (bytes) = sophistication proxy", fontsize=12)
    ax.set_title(
        "Sophistication Proxy Over Time\n"
        "Two-part code Part 1: 'model' complexity (closest to complextropy)",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{output_dir}/plot6_sophistication.png"
    plt.savefig(outpath, dpi=150)
    print(f"  Saved: {outpath}")
    plt.close()


def plot_animation_frames(results, output_dir='.'):
    """
    Plot 7: Grid at many time points in a single figure -- showing
    the full mixing progression.
    """
    grids = results['grids']
    times = results['times']
    n = len(grids)

    # Pick ~20 frames
    n_frames = min(20, n)
    indices = np.linspace(0, n - 1, n_frames, dtype=int)

    cols = 5
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes_flat = axes.flatten()

    for i, idx in enumerate(indices):
        axes_flat[i].imshow(grids[idx], cmap='gray', interpolation='nearest')
        axes_flat[i].set_title(f"t={times[idx]:,}", fontsize=8)
        axes_flat[i].axis('off')

    # Hide unused axes
    for i in range(len(indices), len(axes_flat)):
        axes_flat[i].axis('off')

    fig.suptitle(
        "Coffee Mixing Progression\n"
        "From separated to complex tendrils to uniform mixing",
        fontsize=13
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot7_progression.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def generate_all_plots(args):
    """Generate all 7 plots."""
    print("=" * 60)
    print("Generating Visualization Suite")
    print("Aaronson's 'First Law of Complexodynamics' (2011)")
    print("=" * 60)
    print()

    print("Running simulation...")
    results = run_simulation(
        grid_size=args.grid_size,
        n_steps=args.steps,
        swaps_per_step=args.swaps_per_step,
        measure_interval=args.measure_interval,
        seed=args.seed,
    )
    print(f"  Grid: {args.grid_size}x{args.grid_size}")
    print(f"  Total swaps: {args.steps * args.swaps_per_step:,}")
    print(f"  Measurements: {len(results['times'])}")
    print()

    print("Generating plots...")
    plot_complexity_hump(results, args.output_dir)
    plot_entropy_vs_complexity(results, args.output_dir)
    plot_grid_snapshots(results, args.output_dir)
    plot_measures_comparison(results, args.output_dir)
    plot_multiscale_heatmap(results, args.output_dir)
    plot_sophistication_trajectory(results, args.output_dir)
    plot_animation_frames(results, args.output_dir)

    print()
    print("All 7 plots generated.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization suite for coffee mixing complexity")
    parser.add_argument('--grid-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--swaps-per-step', type=int, default=50)
    parser.add_argument('--measure-interval', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()

    generate_all_plots(args)
