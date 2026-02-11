"""
Solution 5: End-to-End Deep Speech 2 Pipeline
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class ClippedReLU(nn.Module):
    """min(max(x, 0), 20) — Section 3.1"""
    def forward(self, x):
        return torch.clamp(x, min=0.0, max=20.0)


class MiniDeepSpeech2(nn.Module):
    """A minimal Deep Speech 2 model."""

    def __init__(self, n_freq: int, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        # Conv layer: (batch, 1, freq, time) -> (batch, 32, freq', time)
        self.conv = nn.Conv2d(
            1, 32,
            kernel_size=(21, 11),
            stride=(2, 1),
            padding=(10, 5),
            bias=False
        )
        self.conv_bn = nn.BatchNorm2d(32)
        self.conv_act = ClippedReLU()

        # Frequency dimension after conv
        freq_after = (n_freq + 2 * 10 - 21) // 2 + 1
        rnn_input_size = freq_after * 32

        # Two bidirectional GRU layers
        self.gru1 = nn.GRU(
            rnn_input_size, hidden_size,
            batch_first=True, bidirectional=True
        )
        self.gru2 = nn.GRU(
            hidden_size * 2, hidden_size,
            batch_first=True, bidirectional=True
        )

        # Output FC
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        # Conv + BN + ClippedReLU
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_act(x)

        # Reshape for RNN: (batch, channels, freq', time') -> (batch, time', channels*freq')
        _, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, time, channels * freq)

        # Output lengths (conv stride=1 in time, so length is preserved)
        if lengths is not None:
            # The conv has stride 1 in time with padding 5 and kernel 11
            # so output time = input time (same padding)
            output_lengths = lengths.clone()
        else:
            output_lengths = torch.full(
                (batch_size,), time, dtype=torch.long, device=x.device
            )

        # GRU layers
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, output_lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            x, _ = self.gru1(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            packed = nn.utils.rnn.pack_padded_sequence(
                x, output_lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            x, _ = self.gru2(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)

        # FC + log softmax
        x = self.fc(x)  # (batch, time, vocab)
        log_probs = F.log_softmax(x, dim=-1)

        # CTC expects (time, batch, vocab)
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs, output_lengths


def test_pipeline():
    """Test the end-to-end pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)

    from implementation import (
        CharEncoder, generate_synthetic_dataset,
        collate_batch, greedy_decode, word_error_rate
    )

    encoder = CharEncoder('english')
    train_data = generate_synthetic_dataset(30, min_words=1, max_words=2)
    n_fft = 256

    batch = collate_batch(train_data[:4], encoder, n_fft=n_fft)
    n_freq = batch.features.size(2)

    model = MiniDeepSpeech2(n_freq, encoder.vocab_size, hidden_size=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Forward pass
    log_probs, output_lengths = model(batch.features, batch.feature_lengths)
    assert log_probs.dim() == 3
    assert log_probs.size(1) == 4
    assert log_probs.size(2) == encoder.vocab_size
    print(f"[OK] Forward: {log_probs.shape}")

    # Probabilities sum to 1
    prob_sum = log_probs[0, 0].exp().sum().item()
    assert abs(prob_sum - 1.0) < 0.01
    print(f"[OK] Prob sum: {prob_sum:.4f}")

    # CTC loss
    loss = F.ctc_loss(
        log_probs, batch.targets, output_lengths, batch.target_lengths,
        blank=0, reduction='mean', zero_infinity=True
    )
    assert loss.item() > 0
    print(f"[OK] Loss: {loss.item():.4f}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(3):
        model.train()
        optimizer.zero_grad()
        lp, ol = model(batch.features, batch.feature_lengths)
        l = F.ctc_loss(lp, batch.targets, ol, batch.target_lengths,
                       blank=0, reduction='mean', zero_infinity=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400)
        optimizer.step()
        losses.append(l.item())

    print(f"[OK] Losses: {[f'{l:.2f}' for l in losses]}")

    # Decode
    model.eval()
    with torch.no_grad():
        lp, ol = model(batch.features, batch.feature_lengths)
        decoded = greedy_decode(lp, encoder, ol)
        for i in range(min(3, len(decoded))):
            print(f"  Target: \"{train_data[i].transcript}\" | Decoded: \"{decoded[i]}\"")

    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_pipeline()
