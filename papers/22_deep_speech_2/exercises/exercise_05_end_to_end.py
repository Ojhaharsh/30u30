"""
Exercise 5: End-to-End Deep Speech 2 Pipeline
Difficulty: Hard (4/5) | Time: 60 min

Wire together all the components into a working DS2 model:
    Spectrogram -> Conv -> BiGRU -> FC -> CTC

This exercise ties together everything from exercises 1-4.

Your job:
    1. Build the convolutional front-end
    2. Stack bidirectional GRU layers
    3. Add the output FC layer
    4. Implement the forward pass
    5. Train on synthetic data and measure WER
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class ClippedReLU(nn.Module):
    """min(max(x, 0), 20) — Section 3.1"""
    def forward(self, x):
        return torch.clamp(x, min=0.0, max=20.0)


class MiniDeepSpeech2(nn.Module):
    """A minimal Deep Speech 2 model.

    Architecture (Section 3.1):
        1. One Conv2D layer (extract spectral patterns)
        2. Two Bidirectional GRU layers
        3. One FC layer -> log_softmax

    This is a simplified version for learning. The full model
    in the paper uses up to 3 conv + 7 RNN layers.

    Args:
        n_freq: Number of frequency bins in spectrogram
        vocab_size: Output alphabet size (29 for English)
        hidden_size: GRU hidden size per direction
    """

    def __init__(self, n_freq: int, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        # TODO: Create a single Conv2D layer
        # Input: (batch, 1, freq, time)
        # Kernel: (21, 11) covering freq x time
        # Stride: (2, 1) — downsample frequency, keep time
        # Padding: (10, 5) — same padding approximately
        # Output channels: 32
        # Follow with BatchNorm2d and ClippedReLU
        self.conv = None  # YOUR CODE HERE
        self.conv_bn = None  # YOUR CODE HERE
        self.conv_act = ClippedReLU()

        # TODO: Calculate the frequency dimension after conv
        # freq_after = (n_freq + 2*10 - 21) // 2 + 1
        freq_after = None  # YOUR CODE HERE
        rnn_input_size = freq_after * 32  # channels * freq

        # TODO: Create two bidirectional GRU layers
        # Layer 1: input_size=rnn_input_size, hidden_size=hidden_size
        # Layer 2: input_size=hidden_size*2, hidden_size=hidden_size
        self.gru1 = None  # YOUR CODE HERE
        self.gru2 = None  # YOUR CODE HERE

        # TODO: Create the output FC layer
        # Input: hidden_size * 2 (bidirectional)
        # Output: vocab_size
        self.fc = None  # YOUR CODE HERE

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Spectrograms, shape (batch, 1, freq, time)
            lengths: Time lengths, shape (batch,)

        Returns:
            log_probs: shape (time, batch, vocab_size)
            output_lengths: shape (batch,)

        Steps:
            1. Conv2D + BN + ClippedReLU
            2. Reshape: (batch, channels, freq', time') -> (batch, time', channels*freq')
            3. BiGRU layer 1
            4. BiGRU layer 2
            5. FC -> log_softmax
            6. Transpose to (time, batch, vocab) for CTC
        """
        # TODO: implement the forward pass
        pass  # YOUR CODE HERE


def generate_simple_data(n_samples: int = 50):
    """Generate synthetic audio-text pairs for testing.

    Returns:
        List of (spectrogram, text) tuples
    """
    from implementation import generate_synthetic_dataset, CharEncoder, collate_batch

    encoder = CharEncoder('english')
    dataset = generate_synthetic_dataset(n_samples, min_words=1, max_words=2)
    return dataset, encoder


def test_pipeline():
    """Test the end-to-end pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)

    from implementation import (
        CharEncoder, generate_synthetic_dataset,
        collate_batch, greedy_decode, word_error_rate
    )

    # Setup
    encoder = CharEncoder('english')
    train_data = generate_synthetic_dataset(30, min_words=1, max_words=2)
    n_fft = 256

    # Build batch to determine n_freq
    batch = collate_batch(train_data[:4], encoder, n_fft=n_fft)
    n_freq = batch.features.size(2)

    # Build model
    model = MiniDeepSpeech2(
        n_freq=n_freq,
        vocab_size=encoder.vocab_size,
        hidden_size=64
    )

    assert model.conv is not None, "Conv layer not created"
    assert model.gru1 is not None, "GRU layer 1 not created"
    assert model.gru2 is not None, "GRU layer 2 not created"
    assert model.fc is not None, "FC layer not created"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Forward pass test
    log_probs, output_lengths = model(batch.features, batch.feature_lengths)
    assert log_probs is not None, "Forward pass returned None"
    assert log_probs.dim() == 3, f"Expected 3D output, got {log_probs.dim()}D"
    assert log_probs.size(1) == 4, f"Batch size should be 4, got {log_probs.size(1)}"
    assert log_probs.size(2) == encoder.vocab_size, "Wrong vocab size in output"
    print(f"[OK] Forward pass: {log_probs.shape}")

    # Verify log probabilities sum to ~1 (in log space)
    prob_sum = log_probs[0, 0].exp().sum().item()
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities should sum to 1, got {prob_sum}"
    print(f"[OK] Probability sum: {prob_sum:.4f}")

    # CTC loss test
    loss = F.ctc_loss(
        log_probs, batch.targets, output_lengths, batch.target_lengths,
        blank=0, reduction='mean', zero_infinity=True
    )
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert not torch.isinf(loss), "Loss is inf"
    print(f"[OK] CTC loss: {loss.item():.4f}")

    # Quick training test (3 steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(3):
        model.train()
        optimizer.zero_grad()
        log_probs, output_lengths = model(batch.features, batch.feature_lengths)
        loss = F.ctc_loss(
            log_probs, batch.targets, output_lengths, batch.target_lengths,
            blank=0, reduction='mean', zero_infinity=True
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400)
        optimizer.step()
        losses.append(loss.item())

    print(f"[OK] Training losses: {[f'{l:.2f}' for l in losses]}")
    assert losses[-1] <= losses[0] * 1.5, "Loss did not decrease at all"

    # Decode test
    model.eval()
    with torch.no_grad():
        log_probs, output_lengths = model(batch.features, batch.feature_lengths)
        decoded = greedy_decode(log_probs, encoder, output_lengths)
        for i in range(min(3, len(decoded))):
            print(f"  Target: \"{train_data[i].transcript}\" | Decoded: \"{decoded[i]}\"")

    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_pipeline()
