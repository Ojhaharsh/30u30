"""
solution_02_pooling.py - Solution for Exercise 2

Implements max pooling forward and backward pass.

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""

import numpy as np


def pool_forward(x, pool_size=2, stride=2):
    """
    Max pooling forward pass.

    CS231n: 2x2 max pooling with stride 2 discards 75% of activations.
    Output size: (W - F) / S + 1

    Returns:
        out: Pooled output
        cache: (x, pool_size, stride) for backward pass
    """
    N, C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            window = x[:, :, h_start:h_start + pool_size, w_start:w_start + pool_size]
            out[:, :, i, j] = np.max(window, axis=(2, 3))

    cache = (x, pool_size, stride)
    return out, cache


def pool_backward(dout, cache):
    """
    Max pooling backward pass.

    Routes gradient only to the position that had the max value.
    CS231n: "only routing the gradient to the input that had the
    highest value in the forward pass."
    """
    x, pool_size, stride = cache
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    window = x[n, c, h_start:h_start + pool_size, w_start:w_start + pool_size]
                    # Find position of max value in this window
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    # Route gradient to that position
                    dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]

    return dx


def check():
    """Test the implementation."""
    np.random.seed(42)

    # Test 1: Forward shape
    x = np.random.randn(2, 3, 8, 8)
    out, cache = pool_forward(x, pool_size=2, stride=2)
    assert out.shape == (2, 3, 4, 4), f"Test 1 FAILED: expected (2,3,4,4), got {out.shape}"
    print("[OK] Test 1: Shape (2,3,4,4) for 8x8 input with 2x2 pooling")

    # Test 2: Max values correct
    x = np.array([[[[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]]]]).astype(float)
    out, cache = pool_forward(x, pool_size=2, stride=2)
    expected = np.array([[[[6, 8], [14, 16]]]]).astype(float)
    assert np.allclose(out, expected), f"Test 2 FAILED: expected {expected}, got {out}"
    print("[OK] Test 2: Max values correct for known input")

    # Test 3: Backward pass
    dout = np.ones_like(out)
    dx = pool_backward(dout, cache)
    assert dx.shape == x.shape, f"Test 3 FAILED: dx shape {dx.shape} != x shape {x.shape}"
    assert dx[0, 0, 0, 1] == 0, "Test 3 FAILED: non-max position should have 0 gradient"
    assert dx[0, 0, 1, 1] == 1, "Test 3 FAILED: max position (6) should have gradient 1"
    print("[OK] Test 3: Backward pass routes gradients correctly")

    # Test 4: Gradient conservation
    x = np.random.randn(1, 1, 4, 4)
    out, cache = pool_forward(x, pool_size=2, stride=2)
    dout = np.random.randn(*out.shape)
    dx = pool_backward(dout, cache)
    assert np.sum(np.abs(dx) > 0) == out.size, \
        f"Test 4 FAILED: number of nonzero gradients should equal output size"
    print("[OK] Test 4: Gradient conservation")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
