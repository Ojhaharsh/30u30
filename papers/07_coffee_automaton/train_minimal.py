"""
train_minimal.py — Run the coffee automaton experiments from the command line.

Usage:
    # Basic run: interacting model, shows entropy + complexity over time
    python train_minimal.py --grid-size 50 --steps 100000

    # Compare interacting vs non-interacting (replicates Figures 2 and 10)
    python train_minimal.py --grid-size 50 --steps 100000 --compare-models

    # Adjusted coarse-graining (Section 6) vs basic (Section 5)
    python train_minimal.py --grid-size 50 --steps 100000 --compare-methods

    # Scaling experiment (replicates Figures 6-8)
    python train_minimal.py --scaling --sizes 10 20 30 50

    # Non-interacting model only
    python train_minimal.py --grid-size 50 --steps 50000 --model non_interacting

Paper: Aaronson, Carroll, Ouellette (2014), arXiv:1405.6903
"""

import argparse
import sys
import os
import numpy as np

# Add parent dir to path so we can import implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    CoffeeAutomaton, run_simulation, compare_models,
    scaling_experiment, measure_entropy, measure_complexity
)


def plot_single_run(results, model_name='interacting', save_path=None):
    """Plot entropy and complexity vs time for a single simulation."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Entropy and complexity on same time axis
    ax1.plot(results['times'], results['entropy'], 'b-', label='Entropy (gzip fine-grained)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Compressed size (bytes)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'{model_name.replace("_", "-").title()} Model')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(results['times'], results['complexity'], 'r-', label='Complexity (gzip coarse-grained)')
    ax1_twin.set_ylabel('Complexity (bytes)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # Complexity alone with peak marked
    ax2.plot(results['times'], results['complexity'], 'r-', linewidth=2)
    peak_idx = np.argmax(results['complexity'])
    ax2.axvline(results['times'][peak_idx], color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'Peak at t={results["times"][peak_idx]}',
                 xy=(results['times'][peak_idx], results['complexity'][peak_idx]),
                 xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Apparent complexity (bytes)')
    ax2.set_title('Complexity Rise and Fall')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_model_comparison(interacting, non_interacting, save_path=None):
    """
    Compare interacting vs non-interacting — replicates Figure 2 / Figure 10.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Interacting entropy
    axes[0, 0].plot(interacting['times'], interacting['entropy'], 'b-')
    axes[0, 0].set_title('Interacting: Entropy')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('gzip size (bytes)')

    # Interacting complexity
    axes[0, 1].plot(interacting['times'], interacting['complexity'], 'r-')
    axes[0, 1].set_title('Interacting: Complexity')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('gzip size (bytes)')

    # Non-interacting entropy
    axes[1, 0].plot(non_interacting['times'], non_interacting['entropy'], 'b-')
    axes[1, 0].set_title('Non-Interacting: Entropy')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('gzip size (bytes)')

    # Non-interacting complexity
    axes[1, 1].plot(non_interacting['times'], non_interacting['complexity'], 'r-')
    axes[1, 1].set_title('Non-Interacting: Complexity')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('gzip size (bytes)')

    plt.suptitle('Coffee Automaton: Interacting vs Non-Interacting\n'
                 '(Aaronson, Carroll, Ouellette 2014 — Figures 2/10)',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_method_comparison(results_3bucket, results_7bucket, save_path=None):
    """
    Compare Section 5 (3-bucket) vs Section 6 (7-bucket adjusted).
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results_3bucket['times'], results_3bucket['complexity'],
             'r-', label='3-bucket (Section 5)')
    ax1.plot(results_7bucket['times'], results_7bucket['complexity'],
             'b-', label='7-bucket adjusted (Section 6)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Apparent complexity (bytes)')
    ax1.set_title('Interacting Model')
    ax1.legend()

    # If we have non-interacting data...
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Apparent complexity (bytes)')
    ax2.set_title('Non-Interacting Model')
    ax2.text(0.5, 0.5, 'Run with --compare-models\nfor non-interacting data',
             ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='gray')

    plt.suptitle('Section 5 vs Section 6 Coarse-Graining Methods')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling(scaling_results, save_path=None):
    """
    Plot scaling results — replicates Figures 6-8.
    """
    import matplotlib.pyplot as plt

    sizes = scaling_results['sizes']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Max entropy vs n (should be quadratic ~ n^2)
    axes[0].plot(sizes, scaling_results['max_entropy'], 'bo-')
    # Quadratic fit
    coeffs = np.polyfit(sizes, scaling_results['max_entropy'], 2)
    x_fit = np.linspace(sizes[0], sizes[-1], 50)
    axes[0].plot(x_fit, np.polyval(coeffs, x_fit), 'b--', alpha=0.5,
                 label=f'Quadratic fit')
    axes[0].set_xlabel('Grid size (n)')
    axes[0].set_ylabel('Max entropy (bytes)')
    axes[0].set_title('Max Entropy vs Grid Size\n(Expected: ~ n^2)')
    axes[0].legend()

    # Max complexity vs n (should be linear ~ n)
    axes[1].plot(sizes, scaling_results['max_complexity'], 'ro-')
    # Linear fit
    coeffs = np.polyfit(sizes, scaling_results['max_complexity'], 1)
    axes[1].plot(x_fit, np.polyval(coeffs, x_fit), 'r--', alpha=0.5,
                 label=f'Linear fit')
    axes[1].set_xlabel('Grid size (n)')
    axes[1].set_ylabel('Max complexity (bytes)')
    axes[1].set_title('Max Complexity vs Grid Size\n(Expected: ~ n)')
    axes[1].legend()

    # Time to max complexity vs n (should be quadratic ~ n^2)
    axes[2].plot(sizes, scaling_results['peak_time'], 'go-')
    coeffs = np.polyfit(sizes, scaling_results['peak_time'], 2)
    axes[2].plot(x_fit, np.polyval(coeffs, x_fit), 'g--', alpha=0.5,
                 label=f'Quadratic fit')
    axes[2].set_xlabel('Grid size (n)')
    axes[2].set_ylabel('Time to peak complexity')
    axes[2].set_title('Time to Peak vs Grid Size\n(Expected: ~ n^2)')
    axes[2].legend()

    plt.suptitle('Scaling Analysis (Figures 6-8 from Paper)', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Coffee Automaton — Aaronson, Carroll, Ouellette (2014)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_minimal.py --grid-size 50 --steps 100000
  python train_minimal.py --compare-models --grid-size 50 --steps 100000
  python train_minimal.py --scaling --sizes 10 20 30 50
        """
    )

    parser.add_argument('--grid-size', type=int, default=50,
                        help='Grid side length N (default: 50)')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Total simulation steps (default: 100000)')
    parser.add_argument('--snapshots', type=int, default=100,
                        help='Number of measurement points (default: 100)')
    parser.add_argument('--model', type=str, default='interacting',
                        choices=['interacting', 'non_interacting'],
                        help='Which model to run (default: interacting)')
    parser.add_argument('--buckets', type=int, default=7,
                        choices=[3, 7],
                        help='Threshold buckets: 3 (Section 5) or 7 (Section 6)')
    parser.add_argument('--no-adjust', action='store_true',
                        help='Disable row-majority adjustment (use raw thresholding)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Mode flags
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare interacting vs non-interacting')
    parser.add_argument('--compare-methods', action='store_true',
                        help='Compare Section 5 (3-bucket) vs Section 6 (7-bucket)')
    parser.add_argument('--scaling', action='store_true',
                        help='Run scaling experiment (Figures 6-8)')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='Grid sizes for scaling experiment')

    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (print results only)')

    args = parser.parse_args()

    if args.scaling:
        # Scaling experiment (Figures 6-8)
        print("Scaling experiment (Figures 6-8 from paper)")
        print("=" * 50)
        sizes = args.sizes or [10, 20, 30, 50]
        results = scaling_experiment(sizes=sizes, seed=args.seed)

        print("\nResults:")
        print(f"  {'Size':>6} {'Max Entropy':>12} {'Max Complexity':>14} {'Peak Time':>10}")
        print(f"  {'----':>6} {'-----------':>12} {'--------------':>14} {'---------':>10}")
        for i, n in enumerate(results['sizes']):
            print(f"  {n:>6} {results['max_entropy'][i]:>12} "
                  f"{results['max_complexity'][i]:>14} {results['peak_time'][i]:>10}")

        if not args.no_plot:
            plot_scaling(results, save_path=args.save)

    elif args.compare_models:
        # Interacting vs non-interacting (Figures 2 / 10)
        print("Comparing interacting vs non-interacting models")
        print("=" * 50)
        inter, non_inter = compare_models(
            args.grid_size, args.steps, args.snapshots, args.seed)

        inter_peak = np.argmax(inter['complexity'])
        non_peak = np.argmax(non_inter['complexity'])
        print(f"\nInteracting: peak complexity = {inter['complexity'][inter_peak]} "
              f"at t = {inter['times'][inter_peak]}")
        print(f"Non-interacting: peak complexity = {non_inter['complexity'][non_peak]} "
              f"at t = {non_inter['times'][non_peak]}")

        if not args.no_plot:
            plot_model_comparison(inter, non_inter, save_path=args.save)

    elif args.compare_methods:
        # Section 5 vs Section 6 methods
        print("Comparing coarse-graining methods")
        print("=" * 50)

        print("Running with 3-bucket thresholding (Section 5)...")
        results_3 = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, num_buckets=3, adjust=False, seed=args.seed)

        print("Running with 7-bucket adjusted (Section 6)...")
        results_7 = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, num_buckets=7, adjust=True, seed=args.seed)

        peak_3 = np.argmax(results_3['complexity'])
        peak_7 = np.argmax(results_7['complexity'])
        print(f"\n3-bucket: peak = {results_3['complexity'][peak_3]} at t={results_3['times'][peak_3]}")
        print(f"7-bucket adjusted: peak = {results_7['complexity'][peak_7]} at t={results_7['times'][peak_7]}")

        if not args.no_plot:
            plot_method_comparison(results_3, results_7, save_path=args.save)

    else:
        # Single model run
        print(f"Running {args.model} model ({args.grid_size}x{args.grid_size}, "
              f"{args.steps} steps, {args.buckets} buckets, "
              f"adjust={'off' if args.no_adjust else 'on'})")
        print("=" * 50)

        results = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, num_buckets=args.buckets,
            adjust=not args.no_adjust, seed=args.seed)

        peak_idx = np.argmax(results['complexity'])
        print(f"\nResults:")
        print(f"  Peak complexity: {results['complexity'][peak_idx]} bytes "
              f"at t = {results['times'][peak_idx]}")
        print(f"  Final entropy: {results['entropy'][-1]} bytes")
        print(f"  Final complexity: {results['complexity'][-1]} bytes")
        print(f"  Mixing fraction: {results['fraction_mixed'][-1]:.3f}")

        if not args.no_plot:
            plot_single_run(results, args.model, save_path=args.save)


if __name__ == '__main__':
    main()
