"""
Solution 3: RNN with Sequence-wise Batch Normalization
"""

import torch
import torch.nn as nn
from typing import Optional


class SequenceBatchNorm(nn.Module):
    """Sequence-wise Batch Normalization (Section 3.2)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Compute stats over (batch * time, features)
            if x.dim() == 3:
                flat = x.reshape(-1, self.num_features)
            else:
                flat = x

            mean = flat.mean(dim=0)
            var = flat.var(dim=0, unbiased=False)

            # Update running stats
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class BidirectionalGRULayer(nn.Module):
    """Bidirectional GRU with sequence-wise BatchNorm and ClippedReLU."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size, hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.bn = SequenceBatchNorm(hidden_size * 2)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        else:
            output, _ = self.gru(x)

        # Sequence-wise BatchNorm
        output = self.bn(output)

        # Clipped ReLU: min(max(x, 0), 20)
        output = torch.clamp(output, min=0.0, max=20.0)

        return output


def test_rnn_batchnorm():
    """Test the implementation."""
    torch.manual_seed(42)

    # Test SequenceBatchNorm
    bn = SequenceBatchNorm(64)
    x = torch.randn(4, 10, 64)

    bn.train()
    out = bn(x)
    assert out.shape == x.shape
    flat = out.reshape(-1, 64)
    mean = flat.mean(dim=0)
    assert mean.abs().max() < 0.5
    print(f"[OK] SequenceBatchNorm: {out.shape}, mean~0")

    bn.eval()
    out_eval = bn(x)
    assert out_eval is not None
    print(f"[OK] Eval mode works")

    # Test BidirectionalGRULayer
    layer = BidirectionalGRULayer(32, 64)
    x2 = torch.randn(4, 10, 32)
    lengths = torch.tensor([10, 8, 6, 4])

    layer.train()
    out2 = layer(x2, lengths)
    assert out2.shape == (4, 10, 128)
    assert out2.min() >= 0
    assert out2.max() <= 20.0
    print(f"[OK] BiGRU: {out2.shape}, range [{out2.min():.2f}, {out2.max():.2f}]")

    out3 = layer(x2)
    assert out3.shape == (4, 10, 128)
    print(f"[OK] Works without lengths")
    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_rnn_batchnorm()
