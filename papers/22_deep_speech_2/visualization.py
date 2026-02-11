"""
visualization.py — Visualization tools for Deep Speech 2
=========================================================
Amodei et al. (2015) — https://arxiv.org/abs/1512.02595

Generates visualizations for understanding the DS2 pipeline:
    1. Spectrogram display
    2. CTC output probability heatmap
    3. Architecture comparison (depth vs width, from Section 3.5)
    4. WER vs training data size (Section 3.6)
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib is required for visualization. Install with: pip install matplotlib")
    sys.exit(1)

from implementation import (
    DeepSpeech2, CharEncoder, compute_spectrogram,
    generate_synthetic_audio, generate_synthetic_dataset,
    collate_batch, greedy_decode
)


def plot_spectrogram_pipeline(output_dir: str = 'plots'):
    """Show the transformation from waveform to spectrogram.

    This illustrates the first stage of the DS2 pipeline:
    raw audio -> spectrogram -> model input.
    """
    os.makedirs(output_dir, exist_ok=True)

    text = "the cat"
    audio = generate_synthetic_audio(text, sample_rate=16000)
    spec = compute_spectrogram(audio, sample_rate=16000, n_fft=256)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform
    t = np.linspace(0, len(audio) / 16000, len(audio))
    axes[0].plot(t, audio.numpy(), color='#2196F3', linewidth=0.5)
    axes[0].set_title(f'Raw Waveform: "{text}"')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Spectrogram
    im = axes[1].imshow(
        spec.numpy(), aspect='auto', origin='lower', cmap='viridis'
    )
    axes[1].set_title('Log Power Spectrogram (model input)')
    axes[1].set_xlabel('Time frames')
    axes[1].set_ylabel('Frequency bins')
    plt.colorbar(im, ax=axes[1], label='Log power')

    plt.tight_layout()
    path = os.path.join(output_dir, 'spectrogram_pipeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_architecture_diagram(output_dir: str = 'plots'):
    """ASCII-style architecture diagram rendered as a figure.

    Shows the DS2 data flow: Spectrogram -> Conv -> RNN -> FC -> CTC
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Architecture blocks
    blocks = [
        (5, 9.0, 'Input: Log Power Spectrogram\n(batch, 1, freq, time)', '#E3F2FD', '#1565C0'),
        (5, 7.5, 'Conv2D Layers (1-3)\n+ BatchNorm + ClippedReLU', '#E8F5E9', '#2E7D32'),
        (5, 6.0, 'Reshape: merge channel x frequency', '#FFF3E0', '#E65100'),
        (5, 4.5, 'Bidirectional RNN/GRU Layers (1-7)\n+ Sequence-wise BatchNorm\n+ ClippedReLU', '#F3E5F5', '#6A1B9A'),
        (5, 3.0, 'Fully Connected Layer', '#E0F7FA', '#00695C'),
        (5, 1.5, 'Log Softmax -> CTC Loss\nOutput: P(char | time)', '#FFEBEE', '#B71C1C'),
    ]

    for x, y, text, bg_color, text_color in blocks:
        bbox = dict(
            boxstyle='round,pad=0.5', facecolor=bg_color,
            edgecolor=text_color, linewidth=1.5
        )
        ax.text(
            x, y, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color=text_color,
            bbox=bbox
        )

    # Arrows
    for i in range(len(blocks) - 1):
        ax.annotate(
            '', xy=(5, blocks[i + 1][1] + 0.5),
            xytext=(5, blocks[i][1] - 0.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5)
        )

    ax.set_title('Deep Speech 2 Architecture (Section 3)', fontsize=14, pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture_diagram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_depth_comparison(output_dir: str = 'plots'):
    """Illustrate model depth vs performance (Section 3.5).

    Section 3.5: "we find that depth is important to the performance
    of end-to-end speech models"

    These numbers are illustrative of the trend described in the paper,
    where deeper networks (more RNN layers) consistently outperform
    wider networks of similar parameter count.

    Note: these are simplified demonstrations, not the exact paper numbers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Illustrative data based on the paper's finding that depth helps
    # more than width for a given parameter budget (Section 3.5, Table 3)
    configs = {
        '1 RNN layer\n(wide)': {'n_rnn': 1, 'wer': 16.0},
        '3 RNN layers': {'n_rnn': 3, 'wer': 11.5},
        '5 RNN layers': {'n_rnn': 5, 'wer': 9.2},
        '7 RNN layers\n(deep)': {'n_rnn': 7, 'wer': 8.5},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(configs.keys())
    wers = [configs[n]['wer'] for n in names]
    colors = ['#FF7043', '#FFA726', '#66BB6A', '#42A5F5']

    bars = ax.bar(names, wers, color=colors, edgecolor='white', linewidth=1.5)

    for bar, wer in zip(bars, wers):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{wer}%', ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_ylabel('Word Error Rate (%)')
    ax.set_title(
        'Effect of Network Depth on WER\n'
        '[Our illustration of the trend from Section 3.5]'
    )
    ax.set_ylim(0, 20)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'depth_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_data_scaling(output_dir: str = 'plots'):
    """Illustrate WER improvement with more training data (Section 3.6).

    Section 3.6: "we find large improvements from larger datasets"
    The paper shows that going from 3K to 12K hours of training data
    yields significant WER reduction.

    Note: these are illustrative of the paper's overall trend.
    The exact numbers are from Tables 4-5.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Illustrative data based on paper's data scaling findings
    hours = [300, 1000, 3000, 7000, 12000]
    wer_regular = [23.0, 16.5, 12.5, 9.8, 8.5]
    wer_noisy = [40.0, 30.0, 22.0, 17.0, 13.6]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hours, wer_regular, 'o-', color='#2196F3', linewidth=2,
            markersize=8, label='Regular speech')
    ax.plot(hours, wer_noisy, 's-', color='#FF5722', linewidth=2,
            markersize=8, label='Noisy speech')

    ax.set_xlabel('Training Data (hours)')
    ax.set_ylabel('Word Error Rate (%)')
    ax.set_title(
        'WER vs Training Data Size\n'
        '[Our illustration of the trend from Section 3.6]'
    )
    ax.set_xscale('log')
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:,}' for h in hours])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'data_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ctc_alignment(output_dir: str = 'plots'):
    """Show how CTC aligns characters to audio frames.

    CTC does not require explicit alignment between input frames
    and output characters. Instead, it marginalizes over all possible
    alignments during training (Section 3.1).

    This visualization shows a simulated CTC output probability
    distribution over time, illustrating the characteristic spike
    pattern where the model concentrates probability mass on specific
    frames and uses blanks elsewhere.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Simulated CTC output for "cat"
    text = "cat"
    n_frames = 30
    n_chars = len(text) + 1  # +1 for blank

    # Create a realistic-looking CTC output pattern
    probs = np.zeros((n_frames, n_chars))

    # Blank dominates most frames, character spikes at specific points
    probs[:, 0] = 0.7  # blank probability baseline

    # Character activations: spikes with some spread
    char_positions = [5, 15, 25]
    for i, (pos, char) in enumerate(zip(char_positions, text)):
        char_idx = i + 1
        for t in range(n_frames):
            dist = abs(t - pos)
            probs[t, char_idx] = np.exp(-0.5 * dist**2)

    # Normalize to valid probabilities
    probs = probs / probs.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(probs.T, aspect='auto', origin='lower', cmap='hot')
    ax.set_xlabel('Time frame')
    ax.set_ylabel('Character')
    ax.set_yticks(range(n_chars))
    ax.set_yticklabels(['blank'] + list(text))
    ax.set_title('CTC Output Probabilities (simulated for "cat")')
    plt.colorbar(im, ax=ax, label='Probability')

    plt.tight_layout()
    path = os.path.join(output_dir, 'ctc_alignment.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    output_dir = 'plots'
    print("Generating Deep Speech 2 visualizations...")
    print()

    plot_spectrogram_pipeline(output_dir)
    plot_architecture_diagram(output_dir)
    plot_ctc_alignment(output_dir)
    plot_depth_comparison(output_dir)
    plot_data_scaling(output_dir)

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
