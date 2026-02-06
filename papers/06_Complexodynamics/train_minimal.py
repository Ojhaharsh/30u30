"""
train_minimal.py - Coffee Mixing Simulation CLI

Run the coffee-milk mixing simulation from Aaronson's "First Law of
Complexodynamics" blog post and visualize the complexity hump.

Usage:
    # Basic run
    python train_minimal.py --grid-size 64 --steps 50000

    # Compare complexity measures
    python train_minimal.py --grid-size 64 --steps 50000 --compare-measures

    # Multi-scale analysis (Trevisan's idea)
    python train_minimal.py --grid-size 64 --steps 50000 --multiscale

    # Quick test
    python train_minimal.py --grid-size 32 --steps 5000

Reference: https://scottaaronson.blog/?p=762
"""

import argparse
import sys
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[NOTE] matplotlib not installed. Plots will be skipped.")
    print("       Install with: pip install matplotlib")

from implementation import (
    create_initial_grid,
    batch_swap,
    gzip_complexity,
    coarse_grained_kc,
    two_part_code,
    grid_entropy,
    mean_local_entropy,
    boundary_fraction,
    multiscale_kc,
    robust_gzip_complexity,
    sophistication_proxy,
    summarize_results,
    run_simulation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coffee Mixing Simulation: The First Law of Complexodynamics"
    )
    parser.add_argument(
        '--grid-size', type=int, default=64,
        help='Side length of square grid (default: 64)'
    )
    parser.add_argument(
        '--steps', type=int, default=50000,
        help='Number of batch-swap steps (default: 50000)'
    )
    parser.add_argument(
        '--swaps-per-step', type=int, default=50,
        help='Individual swaps per batch step (default: 50)'
    )
    parser.add_argument(
        '--measure-interval', type=int, default=500,
        help='Steps between complexity measurements (default: 500)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--compare-measures', action='store_true',
        help='Compare all complexity measures on same simulation'
    )
    parser.add_argument(
        '--multiscale', action='store_true',
        help='Run multi-scale analysis (Trevisan, blog comment #17)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip plotting (just print results)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help='Directory for output files (default: current directory)'
    )
    return parser.parse_args()


def run_basic(args):
    """Run basic simulation and plot the complexity hump."""
    print(f"Running coffee mixing simulation...")
    print(f"  Grid: {args.grid_size}x{args.grid_size}")
    print(f"  Steps: {args.steps:,} (x{args.swaps_per_step} swaps each)")
    print(f"  Total swaps: {args.steps * args.swaps_per_step:,}")
    print()

    start_time = time.time()

    results = run_simulation(
        grid_size=args.grid_size,
        n_steps=args.steps,
        swaps_per_step=args.swaps_per_step,
        measure_interval=args.measure_interval,
        seed=args.seed,
    )

    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.1f}s")
    print()
    print(summarize_results(results))

    if HAS_MATPLOTLIB and not args.no_plot:
        plot_basic_results(results, args.output_dir)

    return results


def plot_basic_results(results, output_dir='.'):
    """Plot entropy vs complexity (the core demonstration)."""
    times = results['times']
    entropy = results['entropy']
    gzip_c = results['gzip_complexity']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Entropy (monotone increase via local entropy measure)
    axes[0].plot(times, entropy, 'b-', linewidth=1.5)
    axes[0].set_title("Global Entropy")
    axes[0].set_xlabel("Total swaps")
    axes[0].set_ylabel("Shannon entropy (bits)")
    axes[0].grid(True, alpha=0.3)

    # 2. Complexity (the hump)
    axes[1].plot(times, gzip_c, 'r-', linewidth=1.5)
    peak_idx = np.argmax(gzip_c)
    axes[1].axvline(x=times[peak_idx], color='gray', linestyle='--', alpha=0.5)
    axes[1].plot(times[peak_idx], gzip_c[peak_idx], 'ko', markersize=8)
    axes[1].set_title("Complexity (gzip size) -- THE HUMP")
    axes[1].set_xlabel("Total swaps")
    axes[1].set_ylabel("Compressed size (bytes)")
    axes[1].grid(True, alpha=0.3)

    # 3. Grid snapshots
    n_grids = len(results['grids'])
    snapshot_indices = [0, n_grids // 4, n_grids // 2, -1]
    snapshot_labels = ['t=0\n(separated)', 'early\n(tendrils forming)',
                       'mid\n(complex)', 'late\n(mixed)']
    for i, (idx, label) in enumerate(zip(snapshot_indices, snapshot_labels)):
        ax_inset = fig.add_axes([0.68 + i * 0.08, 0.55, 0.07, 0.35])
        ax_inset.imshow(results['grids'][idx], cmap='gray', interpolation='nearest')
        ax_inset.set_title(label, fontsize=7)
        ax_inset.axis('off')

    axes[2].axis('off')
    axes[2].set_title("Grid Snapshots (see insets)")

    plt.tight_layout()
    outpath = f"{output_dir}/complexity_hump.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
    plt.close()


def run_compare_measures(args):
    """Compare all complexity measures on the same simulation."""
    print("Running simulation with all complexity measures...")
    print()

    results = run_simulation(
        grid_size=args.grid_size,
        n_steps=args.steps,
        swaps_per_step=args.swaps_per_step,
        measure_interval=args.measure_interval,
        seed=args.seed,
    )

    print(summarize_results(results))

    if HAS_MATPLOTLIB and not args.no_plot:
        plot_compare_measures(results, args.output_dir)

    return results


def plot_compare_measures(results, output_dir='.'):
    """Plot all complexity measures side by side."""
    times = results['times']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. gzip complexity
    axes[0, 0].plot(times, results['gzip_complexity'], 'r-', linewidth=1.5)
    axes[0, 0].set_title("gzip Complexity (raw KC proxy)")
    axes[0, 0].set_xlabel("Total swaps")
    axes[0, 0].set_ylabel("Compressed bytes")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Coarse-grained KC at different scales
    for bs, vals in results['coarse_kc'].items():
        if len(vals) > 0 and max(vals) > 0:
            axes[0, 1].plot(times, vals, linewidth=1.5, label=f"block={bs}")
    axes[0, 1].set_title("Coarse-grained KC (Carroll's measure)")
    axes[0, 1].set_xlabel("Total swaps")
    axes[0, 1].set_ylabel("Compressed bytes")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Two-part code
    part1 = [p[0] for p in results['two_part']]
    part2 = [p[1] for p in results['two_part']]
    axes[1, 0].plot(times, part1, 'g-', linewidth=1.5, label='Part 1 (model/sophistication)')
    axes[1, 0].plot(times, part2, 'm-', linewidth=1.5, label='Part 2 (residual)')
    axes[1, 0].plot(times, [p1 + p2 for p1, p2 in zip(part1, part2)],
                    'k--', linewidth=1, label='Total')
    axes[1, 0].set_title("Two-Part Code (sophistication proxy)")
    axes[1, 0].set_xlabel("Total swaps")
    axes[1, 0].set_ylabel("Compressed bytes")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Boundary fraction
    axes[1, 1].plot(times, results['fraction_mixed'], 'b-', linewidth=1.5)
    axes[1, 1].set_title("Boundary Fraction (interface length)")
    axes[1, 1].set_xlabel("Total swaps")
    axes[1, 1].set_ylabel("Fraction of differing neighbor pairs")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Complexity Measures Comparison\n"
        "Aaronson's 'First Law of Complexodynamics' (2011)",
        fontsize=14
    )
    plt.tight_layout()
    outpath = f"{output_dir}/measures_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
    plt.close()


def run_multiscale(args):
    """Run multi-scale analysis a la Trevisan (blog comment #17)."""
    print("Running multi-scale complexity analysis (Trevisan's idea)...")
    print()

    np.random.seed(args.seed)
    grid = create_initial_grid(args.grid_size)

    # Scales to analyze
    max_power = int(np.log2(args.grid_size)) - 1
    scales = [2**k for k in range(1, max_power + 1)]

    # Collect data
    n_measurements = min(100, args.steps // max(args.measure_interval, 1))
    step_interval = args.steps // n_measurements

    time_points = []
    scale_time_kc = {s: [] for s in scales}

    # Initial measurement
    time_points.append(0)
    ms = multiscale_kc(grid, scales)
    for s in scales:
        scale_time_kc[s].append(ms.get(s, 0))

    for i in range(1, n_measurements + 1):
        for _ in range(step_interval):
            batch_swap(grid, n_swaps=args.swaps_per_step)

        time_points.append(i * step_interval * args.swaps_per_step)
        ms = multiscale_kc(grid, scales)
        for s in scales:
            scale_time_kc[s].append(ms.get(s, 0))

        if i % 10 == 0:
            print(f"  Progress: {i}/{n_measurements}")

    print(f"\nMulti-scale analysis complete.")

    if HAS_MATPLOTLIB and not args.no_plot:
        plot_multiscale(time_points, scale_time_kc, scales, args.output_dir)


def plot_multiscale(time_points, scale_time_kc, scales, output_dir='.'):
    """Plot the multi-scale complexity heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Lines: KC at each scale over time
    for s in scales:
        axes[0].plot(time_points, scale_time_kc[s], linewidth=1.5, label=f"scale={s}")
    axes[0].set_title("KC at Different Coarse-Graining Scales")
    axes[0].set_xlabel("Total swaps")
    axes[0].set_ylabel("Compressed bytes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Heatmap: scale x time
    matrix = np.array([scale_time_kc[s] for s in scales])
    im = axes[1].imshow(
        matrix, aspect='auto', cmap='hot', interpolation='nearest',
        extent=[time_points[0], time_points[-1], len(scales) - 0.5, -0.5]
    )
    axes[1].set_yticks(range(len(scales)))
    axes[1].set_yticklabels([str(s) for s in scales])
    axes[1].set_title("Multi-Scale Complexity Heatmap (Trevisan)")
    axes[1].set_xlabel("Total swaps")
    axes[1].set_ylabel("Coarse-graining scale")
    plt.colorbar(im, ax=axes[1], label="Compressed bytes")

    fig.suptitle(
        "Multi-Scale Analysis of Coffee Mixing Complexity\n"
        "Trevisan's suggestion (blog comment #17)",
        fontsize=13
    )
    plt.tight_layout()
    outpath = f"{output_dir}/multiscale_analysis.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    if args.compare_measures:
        run_compare_measures(args)
    elif args.multiscale:
        run_multiscale(args)
    else:
        run_basic(args)
