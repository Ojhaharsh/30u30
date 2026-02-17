"""
solution_01_conv_forward.py - Solution for Exercise 1

Implements the naive convolution forward pass.

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""

import numpy as np


def conv_forward(x, w, b, stride=1, pad=0):
    """
    Naive convolution forward pass.

    Slides each filter across the input volume and computes dot products.
    Output size: (W - F + 2P) / S + 1

    Args:
        x: Input volume, shape (N, C, H, W)
        w: Filters, shape (K, C, FH, FW)
        b: Biases, shape (K,)
        stride: Step size for sliding the filter
        pad: Zero-padding

    Returns:
        out: Output volume, shape (N, K, H_out, W_out)
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape

    # Output size formula from CS231n
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    # Pad input (only on spatial dims)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Initialize output
    out = np.zeros((N, K, H_out, W_out))

    # Compute convolution with nested loops
    for n in range(N):                  # Each image in batch
        for k in range(K):              # Each filter
            for i in range(H_out):      # Each output row
                for j in range(W_out):  # Each output col
                    h_start = i * stride
                    w_start = j * stride
                    # Extract the local region that this filter position covers
                    region = x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                    # Dot product between filter and local region, plus bias
                    out[n, k, i, j] = np.sum(region * w[k]) + b[k]

    return out


def check():
    """Test the implementation."""
    np.random.seed(42)

    # Test 1: Basic convolution
    x = np.random.randn(1, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.zeros(2)
    out = conv_forward(x, w, b, stride=1, pad=0)
    assert out.shape == (1, 2, 3, 3), f"Test 1 FAILED: expected (1,2,3,3), got {out.shape}"
    print("[OK] Test 1: Shape (1,2,3,3) for 5x5 input, 3x3 filter, no padding")

    # Test 2: With padding
    out = conv_forward(x, w, b, stride=1, pad=1)
    assert out.shape == (1, 2, 5, 5), f"Test 2 FAILED: expected (1,2,5,5), got {out.shape}"
    print("[OK] Test 2: Shape (1,2,5,5) with pad=1")

    # Test 3: With stride
    x = np.random.randn(2, 3, 8, 8)
    w = np.random.randn(4, 3, 3, 3)
    b = np.zeros(4)
    out = conv_forward(x, w, b, stride=2, pad=1)
    assert out.shape == (2, 4, 4, 4), f"Test 3 FAILED: expected (2,4,4,4), got {out.shape}"
    print("[OK] Test 3: Shape (2,4,4,4) with stride=2")

    # Test 4: Known computation
    x = np.ones((1, 1, 3, 3))
    w = np.ones((1, 1, 2, 2))
    b = np.array([0.0])
    out = conv_forward(x, w, b, stride=1, pad=0)
    assert out.shape == (1, 1, 2, 2), f"Test 4 FAILED: shape"
    assert np.allclose(out, 4.0), f"Test 4 FAILED: expected 4.0, got {out}"
    print("[OK] Test 4: All-ones input with all-ones 2x2 filter gives 4.0")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
