"""
train_minimal.py — Training script for Deep Speech 2
=====================================================
Amodei et al. (2015) — https://arxiv.org/abs/1512.02595

Trains a scaled-down Deep Speech 2 model on synthetic audio data.
Demonstrates the full training pipeline: spectrogram extraction,
CTC loss, SortaGrad (first epoch), greedy decoding, and WER tracking.

Usage:
    python train_minimal.py
    python train_minimal.py --epochs 10 --n-rnn 3 --rnn-hidden 256
    python train_minimal.py --output-dir plots
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

# Ensure the parent directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementation import (
    DeepSpeech2, CharEncoder, AudioSample, SpectrogramBatch,
    generate_synthetic_dataset, collate_batch, train_step,
    evaluate, greedy_decode, sortagrad_sampler, compute_spectrogram,
    word_error_rate, character_error_rate
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Deep Speech 2 on synthetic data'
    )
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n-train', type=int, default=80,
                        help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=20,
                        help='Number of validation samples')
    parser.add_argument('--n-conv', type=int, default=2,
                        help='Number of conv layers (paper: 1-3)')
    parser.add_argument('--n-rnn', type=int, default=3,
                        help='Number of RNN layers (paper: 1-7)')
    parser.add_argument('--rnn-hidden', type=int, default=256,
                        help='RNN hidden size per direction')
    parser.add_argument('--rnn-type', type=str, default='gru',
                        choices=['gru', 'rnn'],
                        help='RNN cell type (paper finds GRU trains faster)')
    parser.add_argument('--n-fft', type=int, default=256,
                        help='FFT size for spectrogram')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory for output plots')
    parser.add_argument('--no-sortagrad', action='store_true',
                        help='Disable SortaGrad (first-epoch length sorting)')
    return parser.parse_args()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_wers: list,
    val_cers: list,
    output_dir: str
):
    """Save training curves to disk."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(train_losses, label='Train', color='#2196F3')
    axes[0].plot(
        np.linspace(0, len(train_losses) - 1, len(val_losses)),
        val_losses, label='Val', color='#FF5722', marker='o', markersize=4
    )
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('CTC Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # WER curve
    axes[1].plot(val_wers, color='#4CAF50', marker='o', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Word Error Rate')
    axes[1].set_title('Validation WER')
    axes[1].grid(True, alpha=0.3)

    # CER curve
    axes[2].plot(val_cers, color='#9C27B0', marker='o', markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Character Error Rate')
    axes[2].set_title('Validation CER')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {path}")


def plot_predictions(
    model: DeepSpeech2,
    samples: list,
    encoder: CharEncoder,
    device: torch.device,
    output_dir: str,
    n_fft: int = 256
):
    """Visualize model predictions vs ground truth."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    n_show = min(5, len(samples))

    fig, axes = plt.subplots(n_show, 2, figsize=(14, 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i in range(n_show):
            sample = samples[i]
            batch = collate_batch([sample], encoder, n_fft=n_fft)
            batch = batch.to(device)

            log_probs, output_lengths = model(batch.features, batch.feature_lengths)
            decoded = greedy_decode(log_probs, encoder, output_lengths)

            # Plot spectrogram
            spec = batch.features[0, 0].cpu().numpy()
            axes[i, 0].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 0].set_title(f'Spectrogram: "{sample.transcript}"', fontsize=9)
            axes[i, 0].set_ylabel('Freq')

            # Plot CTC output probabilities
            probs = log_probs[:output_lengths[0], 0, :].exp().cpu().numpy()
            axes[i, 1].imshow(probs.T, aspect='auto', origin='lower', cmap='hot')
            axes[i, 1].set_title(
                f'CTC output | Decoded: "{decoded[0]}"', fontsize=9
            )
            axes[i, 1].set_ylabel('Char idx')

    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')

    plt.tight_layout()
    path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction visualization to {path}")


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Character encoder
    encoder = CharEncoder('english')
    print(f"Vocabulary size: {encoder.vocab_size} (26 letters + space + apostrophe + blank)")

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    print(f"  Training: {args.n_train} samples")
    print(f"  Validation: {args.n_val} samples")
    train_data = generate_synthetic_dataset(args.n_train, min_words=1, max_words=3)
    val_data = generate_synthetic_dataset(args.n_val, min_words=1, max_words=3)

    # Compute n_freq from the spectrogram of the first sample
    sample_spec = compute_spectrogram(train_data[0].audio, n_fft=args.n_fft)
    n_freq = sample_spec.size(0)
    print(f"  Spectrogram: {n_freq} frequency bins")

    # Build model
    model = DeepSpeech2(
        n_freq=n_freq,
        vocab_size=encoder.vocab_size,
        n_conv=args.n_conv,
        n_rnn=args.n_rnn,
        rnn_hidden=args.rnn_hidden,
        rnn_type=args.rnn_type,
        use_batchnorm=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: DS2 with {args.n_conv} conv + {args.n_rnn} {args.rnn_type.upper()} layers")
    print(f"  RNN hidden: {args.rnn_hidden} per direction")
    print(f"  Total parameters: {n_params:,}")

    # Optimizer — paper uses SGD with momentum, we use Adam for faster
    # convergence on small data. This is our choice, not the paper's.
    # Note: The paper uses SGD with Nesterov momentum (Section 3.4).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    if not args.no_sortagrad:
        print("  SortaGrad: ON (sorting by length in epoch 0)")

    train_losses = []
    val_losses = []
    val_wers = []
    val_cers = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        # SortaGrad: sort by length in first epoch (Section 3.1)
        if not args.no_sortagrad:
            indices = sortagrad_sampler(train_data, epoch)
        else:
            indices = list(range(len(train_data)))
            np.random.shuffle(indices)

        epoch_losses = []

        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            batch_samples = [train_data[i] for i in batch_indices]

            batch = collate_batch(batch_samples, encoder, n_fft=args.n_fft)
            loss = train_step(model, batch, optimizer, device)
            epoch_losses.append(loss)
            train_losses.append(loss)

        # Validation
        val_metrics = evaluate(model, val_data, encoder, args.batch_size, device, args.n_fft)
        val_losses.append(val_metrics['loss'])
        val_wers.append(val_metrics['wer'])
        val_cers.append(val_metrics['cer'])

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        sort_str = "(sorted)" if epoch == 0 and not args.no_sortagrad else "(random)"

        print(
            f"  Epoch {epoch + 1}/{args.epochs} {sort_str}: "
            f"train_loss={avg_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"WER={val_metrics['wer']:.2%}, CER={val_metrics['cer']:.2%}, "
            f"time={epoch_time:.1f}s"
        )

    # Show sample predictions
    print("\nSample predictions (greedy decoding):")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        for sample in val_data[:5]:
            batch = collate_batch([sample], encoder, n_fft=args.n_fft)
            batch = batch.to(device)
            log_probs, output_lengths = model(batch.features, batch.feature_lengths)
            decoded = greedy_decode(log_probs, encoder, output_lengths)
            print(f'  Target: "{sample.transcript}"')
            print(f'  Output: "{decoded[0]}"')
            print()

    # Save plots
    plot_training_curves(
        train_losses, val_losses, val_wers, val_cers, args.output_dir
    )
    plot_predictions(
        model, val_data[:5], encoder, device, args.output_dir, args.n_fft
    )

    print(f"\nTraining complete.")
    print(f"  Final WER: {val_wers[-1]:.2%}")
    print(f"  Final CER: {val_cers[-1]:.2%}")


if __name__ == '__main__':
    main()
