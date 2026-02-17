"""
exercise_02_pooling.py - Implement Max Pooling

Difficulty: Medium (3/5)

Task: Implement max pooling forward pass AND backward pass.

The forward pass is straightforward: for each pool_size x pool_size window,
take the maximum value. The backward pass routes gradients only to the
position that had the maximum value during the forward pass.

CS231n: "The backward pass for a max(x,y) operation has a simple
interpretation as only routing the gradient to the input that had the
highest value in the forward pass."

Output size formula (no padding for pooling):
    output_size = (W - F) / S + 1

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""

import numpy as np


def pool_forward(x, pool_size=2, stride=2):
    """
    Max pooling forward pass.

    YOUR TASK: Fill in this function.

    For each pool_size x pool_size window, output the maximum value.
    Also store the indices of the max values for the backward pass.

    Args:
        x: Input, shape (N, C, H, W)
        pool_size: Size of the pooling window (square)
        stride: Step size

    Returns:
        out: Pooled output, shape (N, C, H_out, W_out)
        cache: Information needed for backward pass (store x and max positions)

    Hints:
        1. Compute H_out = (H - pool_size) / stride + 1
        2. Loop over output positions
        3. For each position, extract the window and take np.max
        4. Store the index of the max (np.argmax) for the backward pass
    """
    N, C, H, W = x.shape

    # Step 1: Compute output size
    # H_out = ???
    # W_out = ???
    raise NotImplementedError("Compute output size and implement pooling")

    # Step 2: Initialize output and mask for backward pass
    # out = np.zeros(...)
    # max_indices = {}  # Store (n, c, i, j) -> (h_max, w_max)

    # Step 3: Loop and compute max
    # for i in range(H_out):
    #     for j in range(W_out):
    #         h_start = i * stride
    #         w_start = j * stride
    #         window = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
    #         out[:, :, i, j] = np.max(window, axis=(2, 3))

    # cache = (x, pool_size, stride)
    # return out, cache


def pool_backward(dout, cache):
    """
    Max pooling backward pass.

    YOUR TASK: Fill in this function.

    Route the upstream gradient to the position that had the max value
    during the forward pass. All other positions get zero gradient.

    Args:
        dout: Upstream gradient, shape (N, C, H_out, W_out)
        cache: (x, pool_size, stride) from forward pass

    Returns:
        dx: Gradient w.r.t. input, shape (N, C, H, W)

    Hints:
        1. Initialize dx = np.zeros_like(x)
        2. For each output position, find the max position in the window
        3. Route dout[n, c, i, j] to dx at the max position
    """
    x, pool_size, stride = cache
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    raise NotImplementedError("Implement pooling backward pass")

    # dx = np.zeros_like(x)
    # for n in range(N):
    #     for c in range(C):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 h_start = i * stride
    #                 w_start = j * stride
    #                 window = x[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
    #                 max_idx = np.unravel_index(np.argmax(window), window.shape)
    #                 dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]
    # return dx


def check():
    """Test your implementation."""
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
                     [13, 14, 15, 16]]]]).astype(float)  # (1, 1, 4, 4)
    out, cache = pool_forward(x, pool_size=2, stride=2)
    expected = np.array([[[[6, 8], [14, 16]]]]).astype(float)
    assert np.allclose(out, expected), f"Test 2 FAILED: expected {expected}, got {out}"
    print("[OK] Test 2: Max values correct for known input")

    # Test 3: Backward pass
    dout = np.ones_like(out)
    dx = pool_backward(dout, cache)
    assert dx.shape == x.shape, f"Test 3 FAILED: dx shape {dx.shape} != x shape {x.shape}"
    # Gradient should be 1 at max positions, 0 elsewhere
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
    print("[OK] Test 4: Gradient conservation (one gradient per pool window)")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
