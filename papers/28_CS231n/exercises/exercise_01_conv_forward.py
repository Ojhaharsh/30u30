"""
exercise_01_conv_forward.py - Implement Convolution Forward Pass

Difficulty: Easy (2/5)

Task: Implement the naive convolution forward pass using nested loops.
This is the most fundamental CNN operation — sliding a filter across an
input volume and computing dot products at every spatial position.

The output size formula (from CS231n):
    output_size = (W - F + 2P) / S + 1

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""

import numpy as np


def conv_forward(x, w, b, stride=1, pad=0):
    """
    Compute the forward pass for a convolutional layer.

    YOUR TASK: Fill in this function.

    The convolution slides each filter over the input and computes a dot
    product at each spatial position. The result is a 2D activation map
    per filter.

    Args:
        x: Input volume, shape (N, C, H, W)
           N = batch size, C = channels, H = height, W = width
        w: Filters, shape (K, C, FH, FW)
           K = num filters, C = channels, FH = filter height, FW = filter width
        b: Biases, shape (K,)
        stride: Step size for sliding the filter
        pad: Number of zero-padding pixels added to each border

    Returns:
        out: Output volume, shape (N, K, H_out, W_out)

    Hints:
        1. Compute H_out and W_out using the CS231n formula
        2. Pad the input using np.pad(x, (...), mode='constant')
        3. Use 4 nested loops: batch, filter, output_row, output_col
        4. Extract the local region and compute np.sum(region * w[k]) + b[k]
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape

    # Step 1: Compute output dimensions
    # H_out = ???
    # W_out = ???
    raise NotImplementedError("Compute H_out and W_out using (W - F + 2P)/S + 1")

    # Step 2: Pad the input
    # x_padded = ???

    # Step 3: Initialize output
    # out = np.zeros((N, K, H_out, W_out))

    # Step 4: Nested loops — compute convolution
    # for n in range(N):
    #     for k in range(K):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 h_start = ???
    #                 w_start = ???
    #                 region = ???
    #                 out[n, k, i, j] = ???

    # return out


def check():
    """Test your implementation."""
    np.random.seed(42)

    # Test 1: Basic convolution
    x = np.random.randn(1, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.zeros(2)

    out = conv_forward(x, w, b, stride=1, pad=0)
    assert out.shape == (1, 2, 3, 3), f"Test 1 FAILED: expected (1,2,3,3), got {out.shape}"
    print("[OK] Test 1: Shape (1,2,3,3) for 5x5 input, 3x3 filter, no padding")

    # Test 2: With padding (preserves spatial size)
    out = conv_forward(x, w, b, stride=1, pad=1)
    assert out.shape == (1, 2, 5, 5), f"Test 2 FAILED: expected (1,2,5,5), got {out.shape}"
    print("[OK] Test 2: Shape (1,2,5,5) with pad=1 (spatial size preserved)")

    # Test 3: With stride
    x = np.random.randn(2, 3, 8, 8)
    w = np.random.randn(4, 3, 3, 3)
    b = np.zeros(4)
    out = conv_forward(x, w, b, stride=2, pad=1)
    assert out.shape == (2, 4, 4, 4), f"Test 3 FAILED: expected (2,4,4,4), got {out.shape}"
    print("[OK] Test 3: Shape (2,4,4,4) with stride=2")

    # Test 4: Verify a known computation
    x = np.ones((1, 1, 3, 3))
    w = np.ones((1, 1, 2, 2))
    b = np.array([0.0])
    out = conv_forward(x, w, b, stride=1, pad=0)
    assert out.shape == (1, 1, 2, 2), f"Test 4 FAILED: shape"
    assert np.allclose(out, 4.0), f"Test 4 FAILED: ones*ones on 2x2 should give 4.0, got {out}"
    print("[OK] Test 4: All-ones input with all-ones 2x2 filter gives 4.0")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
