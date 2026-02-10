"""
train_minimal.py - MPNN Training CLI

Train a Message Passing Neural Network on synthetic molecular data.
Based on Gilmer et al. 2017, "Neural Message Passing for Quantum Chemistry".

Usage:
    # Default training (edge network + Set2Set, 50 epochs)
    python train_minimal.py

    # Quick test (2 epochs)
    python train_minimal.py --epochs 2

    # Custom architecture
    python train_minimal.py --hidden-dim 128 --n-messages 6 --message-type edge_network

    # Compare message function variants
    python train_minimal.py --compare-messages

    # Use simple sum readout instead of Set2Set
    python train_minimal.py --readout sum

References:
    Section 2: MPNN framework (message passing + readout)
    Section 4: Variants explored (edge network, Set2Set, virtual edges)
    Section 6, Table 2: Results on QM9
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from implementation import (
    MPNN,
    ATOM_TYPES,
    BOND_TYPES,
    generate_dataset,
    batch_graphs,
    train_epoch,
    evaluate,
    summarize_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MPNN on synthetic molecular data (Gilmer et al. 2017)'
    )
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Node embedding dimension (paper uses 200, we default to 64)')
    parser.add_argument('--n-messages', type=int, default=3,
                        help='Number of message passing rounds T (paper uses 6)')
    parser.add_argument('--message-type', type=str, default='edge_network',
                        choices=['simple', 'matrix', 'edge_network'],
                        help='Message function variant (Section 3-4)')
    parser.add_argument('--readout', type=str, default='set2set',
                        choices=['sum', 'set2set'],
                        help='Readout function (Section 4.3)')
    parser.add_argument('--set2set-steps', type=int, default=6,
                        help='Set2Set processing steps M (paper uses 6)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (paper uses 1e-4 with Adam)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (paper uses 20)')
    parser.add_argument('--n-molecules', type=int, default=500,
                        help='Number of synthetic molecules to generate')
    parser.add_argument('--n-targets', type=int, default=3,
                        help='Number of target properties per molecule')

    # Modes
    parser.add_argument('--compare-messages', action='store_true',
                        help='Compare all three message function variants')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory for saving outputs')

    return parser.parse_args()


def train_model(args, message_type=None, readout_type=None, verbose=True):
    """
    Train an MPNN model with the specified configuration.

    Args:
        args: Parsed command-line arguments.
        message_type: Override for message function type.
        readout_type: Override for readout function type.
        verbose: Whether to print progress.

    Returns:
        Tuple of (model, train_losses, test_metrics).
    """
    if message_type is None:
        message_type = args.message_type
    if readout_type is None:
        readout_type = args.readout

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Device: {device}")

    # Generate data
    if verbose:
        print(f"Generating {args.n_molecules} synthetic molecules...")
    dataset = generate_dataset(
        n_molecules=args.n_molecules,
        n_targets=args.n_targets,
        seed=args.seed
    )

    # Split: 80% train, 20% test
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]
    if verbose:
        print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Create model
    model = MPNN(
        node_dim=len(ATOM_TYPES),
        edge_dim=len(BOND_TYPES),
        hidden_dim=args.hidden_dim,
        output_dim=args.n_targets,
        n_messages=args.n_messages,
        message_type=message_type,
        readout_type=readout_type,
        set2set_steps=args.set2set_steps
    ).to(device)

    if verbose:
        print(f"\n{summarize_model(model)}\n")

    # Optimizer (paper uses Adam, Section 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses = []
    test_metrics_history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        loss = train_epoch(model, train_data, optimizer,
                          batch_size=args.batch_size, device=device)
        metrics = evaluate(model, test_data,
                          batch_size=args.batch_size, device=device)

        elapsed = time.time() - start_time
        train_losses.append(loss)
        test_metrics_history.append(metrics)

        if verbose and (epoch % max(1, args.epochs // 10) == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"train_loss={loss:.4f}, "
                  f"test_mae={metrics['mae']:.4f}, "
                  f"time={elapsed:.1f}s")

    return model, train_losses, test_metrics_history


def run_compare_messages(args):
    """
    Compare all three message function variants on the same data.

    This reproduces (in spirit) the ablation study from Section 6,
    though on synthetic data rather than QM9.
    """
    print("Comparing message function variants")
    print("=" * 60)
    print("(Section 3-4: simple, matrix, edge_network)")
    print()

    results = {}
    for msg_type in ['simple', 'matrix', 'edge_network']:
        print(f"\n--- Message type: {msg_type} ---")
        model, losses, metrics = train_model(
            args, message_type=msg_type, verbose=True
        )
        final_mae = metrics[-1]['mae']
        results[msg_type] = {
            'final_mae': final_mae,
            'losses': losses,
            'metrics': metrics
        }

    # Summary table
    print("\n" + "=" * 60)
    print("Summary: Final Test MAE by Message Function Variant")
    print("-" * 40)
    for msg_type, result in results.items():
        print(f"  {msg_type:15s}: MAE = {result['final_mae']:.4f}")
    print()
    print("Note: On real QM9 data, edge_network consistently outperforms")
    print("the other variants (Table 2). On synthetic data, differences")
    print("may be less pronounced.")

    # Save comparison plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for msg_type, result in results.items():
            ax.plot(result['losses'], label=msg_type)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss (MAE)')
        ax.set_title('Message Function Comparison (Gilmer et al. 2017, Sections 3-4)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'message_comparison.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison plot saved to: {output_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")

    return results


def run_basic(args):
    """Run standard training with the specified configuration."""
    print("MPNN Training — Gilmer et al. 2017")
    print("=" * 50)
    print(f"Architecture: {args.message_type} message + {args.readout} readout")
    print(f"Hidden dim: {args.hidden_dim}, Message rounds: {args.n_messages}")
    print()

    model, losses, metrics = train_model(args, verbose=True)

    # Final evaluation
    final = metrics[-1]
    print(f"\nFinal Results:")
    print(f"  Test MAE: {final['mae']:.4f}")
    print(f"  Test MSE: {final['mse']:.4f}")

    # Save training curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        axes[0].plot(losses)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss (MAE)')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)

        # Test MAE curve
        test_maes = [m['mae'] for m in metrics]
        axes[1].plot(test_maes)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test MAE')
        axes[1].set_title('Test MAE')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('MPNN Training (Gilmer et al. 2017)', fontsize=14)
        plt.tight_layout()

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'training_curves.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nTraining curves saved to: {output_path}")
    except ImportError:
        print("\nmatplotlib not available — skipping plot")

    return model


if __name__ == '__main__':
    args = parse_args()

    if args.compare_messages:
        run_compare_messages(args)
    else:
        run_basic(args)
