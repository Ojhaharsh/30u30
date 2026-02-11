"""
Exercise 3: RNN with Sequence-wise Batch Normalization
Difficulty: Medium (3/5) | Time: 45 min

The paper's key architectural insight for training deep RNNs:
use batch normalization where statistics are computed over the
entire (batch x time) dimension, not per-timestep (Section 3.2).

Your job:
    1. Implement SequenceBatchNorm
    2. Build a BidirectionalGRULayer that uses it
    3. Verify it handles variable-length sequences correctly
"""

import torch
import torch.nn as nn
from typing import Optional


class SequenceBatchNorm(nn.Module):
    """Sequence-wise Batch Normalization (Section 3.2).

    Standard BatchNorm computes statistics per-timestep. This fails for
    variable-length sequences because later timesteps have fewer samples
    contributing to the statistics.

    Sequence-wise BatchNorm computes statistics over ALL items in the
    minibatch across ALL timesteps, treating (batch * time) as one
    large batch dimension.

    Args:
        num_features: Number of features per timestep
        eps: Small constant for numerical stability
        momentum: Running stats momentum
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # TODO: Create learnable affine parameters (weight and bias)
        # weight should be initialized to ones, bias to zeros
        self.weight = None  # YOUR CODE HERE
        self.bias = None  # YOUR CODE HERE

        # TODO: Register running mean and running variance as buffers
        # (These are not trainable parameters, but need to be saved/loaded)
        # Hint: use self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, time, features)

        Returns:
            Normalized tensor, same shape as input

        During training:
            1. Reshape x to (batch * time, features)
            2. Compute mean and variance over dimension 0
            3. Update running_mean and running_var
            4. Normalize x
            5. Apply affine: x_norm * weight + bias

        During eval:
            Use running_mean and running_var instead of batch stats
        """
        # TODO: implement sequence-wise batch normalization
        pass  # YOUR CODE HERE


class BidirectionalGRULayer(nn.Module):
    """A single bidirectional GRU layer with sequence-wise BatchNorm.

    Section 3.1: "bidirectional recurrent layers"
    Section 3.2: "we apply batch normalization on the output activations"

    Architecture: BiGRU -> SequenceBatchNorm -> ClippedReLU

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden size per direction (output will be hidden_size * 2)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # TODO: Create a bidirectional GRU
        # Hint: nn.GRU(..., bidirectional=True, batch_first=True)
        self.gru = None  # YOUR CODE HERE

        # TODO: Create SequenceBatchNorm for the GRU output
        # Note: bidirectional GRU output has size hidden_size * 2
        self.bn = None  # YOUR CODE HERE

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: shape (batch, time, features)
            lengths: actual sequence lengths, shape (batch,)

        Returns:
            shape (batch, time, hidden_size * 2)

        Steps:
            1. If lengths provided, pack the padded sequence
            2. Run through GRU
            3. If packed, unpack
            4. Apply SequenceBatchNorm
            5. Apply ClippedReLU: min(max(x, 0), 20)
        """
        # TODO: implement the forward pass
        pass  # YOUR CODE HERE


def test_rnn_batchnorm():
    """Test your implementation."""
    torch.manual_seed(42)

    # Test SequenceBatchNorm
    bn = SequenceBatchNorm(64)
    x = torch.randn(4, 10, 64)  # (batch=4, time=10, features=64)

    bn.train()
    out = bn(x)
    assert out is not None, "SequenceBatchNorm returned None"
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    # Check that output is approximately normalized
    flat = out.reshape(-1, 64)
    mean = flat.mean(dim=0)
    assert mean.abs().max() < 0.5, f"Mean not near 0: max={mean.abs().max():.4f}"
    print(f"[OK] SequenceBatchNorm output shape: {out.shape}")
    print(f"[OK] Output mean (should be ~0): {mean.abs().mean():.4f}")

    # Test eval mode uses running stats
    bn.eval()
    out_eval = bn(x)
    assert out_eval is not None, "SequenceBatchNorm returned None in eval mode"
    print(f"[OK] SequenceBatchNorm eval mode works")

    # Test BidirectionalGRULayer
    layer = BidirectionalGRULayer(input_size=32, hidden_size=64)
    x2 = torch.randn(4, 10, 32)
    lengths = torch.tensor([10, 8, 6, 4])

    layer.train()
    out2 = layer(x2, lengths)
    assert out2 is not None, "BidirectionalGRULayer returned None"
    assert out2.shape == (4, 10, 128), f"Expected (4,10,128), got {out2.shape}"
    print(f"[OK] BidirectionalGRULayer output shape: {out2.shape}")

    # Check clipped ReLU: all values should be in [0, 20]
    assert out2.min() >= 0, f"ClippedReLU failed: min={out2.min():.4f}"
    assert out2.max() <= 20.0, f"ClippedReLU failed: max={out2.max():.4f}"
    print(f"[OK] ClippedReLU range: [{out2.min():.4f}, {out2.max():.4f}]")

    # Test without lengths (no packing)
    out3 = layer(x2)
    assert out3.shape == (4, 10, 128), f"Expected (4,10,128), got {out3.shape}"
    print(f"[OK] Works without lengths too")

    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_rnn_batchnorm()
