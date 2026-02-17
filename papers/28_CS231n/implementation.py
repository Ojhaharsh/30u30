"""
implementation.py - CNN from Scratch in NumPy

Implements the core convolutional neural network layers described in
Stanford CS231n (https://cs231n.github.io/convolutional-networks/):
  - Convolutional layer (naive + im2col)
  - Max pooling layer
  - ReLU activation
  - Fully connected layer
  - SimpleCNN class (CONV-RELU-POOL-FC pipeline)

Usage:
    python implementation.py
    python implementation.py --demo

Reference: CS231n Course Notes - https://cs231n.github.io/convolutional-networks/
"""

import numpy as np
import argparse


# ---------------------------------------------------------------------------
# Convolutional Layer
# ---------------------------------------------------------------------------

def conv_forward_naive(x, w, b, stride=1, pad=0):
    """
    Naive convolution forward pass using nested loops.

    Slides each filter across the input volume and computes dot products.
    Output size per CS231n: (W - F + 2P) / S + 1

    Args:
        x: Input volume, shape (N, C, H, W) — batch, channels, height, width
        w: Filters, shape (K, C, FH, FW) — num_filters, channels, filter_h, filter_w
        b: Biases, shape (K,)
        stride: Step size for sliding the filter
        pad: Zero-padding added to input borders

    Returns:
        out: Output volume, shape (N, K, H_out, W_out)
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape

    # Output size formula from CS231n
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    out = np.zeros((N, K, H_out, W_out))

    for n in range(N):                  # Each image in batch
        for k in range(K):              # Each filter
            for i in range(H_out):      # Each output row
                for j in range(W_out):  # Each output col
                    h_start = i * stride
                    w_start = j * stride
                    # Extract local region and compute dot product
                    # This is exactly the operation CS231n describes:
                    # "compute dot products between the entries of the filter
                    # and the input at any position"
                    region = x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                    out[n, k, i, j] = np.sum(region * w[k]) + b[k]

    return out


def im2col(x, FH, FW, stride=1, pad=0):
    """
    Reshape input patches into columns for matrix-multiply convolution.

    CS231n describes this trick:
    "The local regions in the input image are stretched out into columns
    in an operation commonly called im2col."

    Args:
        x: Input, shape (N, C, H, W)
        FH, FW: Filter height and width
        stride: Stride
        pad: Padding

    Returns:
        cols: Columns matrix, shape (C*FH*FW, N*H_out*W_out)
    """
    N, C, H, W = x.shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    cols = np.zeros((C * FH * FW, N * H_out * W_out))

    col_idx = 0
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                cols[:, col_idx] = patch.ravel()
                col_idx += 1

    return cols


def conv_forward_im2col(x, w, b, stride=1, pad=0):
    """
    Convolution via im2col + matrix multiply.

    CS231n: "The result of a convolution is now equivalent to performing
    one large matrix multiply np.dot(W_row, X_col)."

    This is much faster than the naive version because it leverages
    optimized BLAS routines for the matrix multiply.

    Args:
        x: Input volume, shape (N, C, H, W)
        w: Filters, shape (K, C, FH, FW)
        b: Biases, shape (K,)
        stride, pad: As in conv_forward_naive

    Returns:
        out: Output volume, shape (N, K, H_out, W_out)
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    # Stretch input patches into columns
    cols = im2col(x, FH, FW, stride, pad)

    # Reshape filters into rows: (K, C*FH*FW)
    w_row = w.reshape(K, -1)

    # Matrix multiply: (K, C*FH*FW) @ (C*FH*FW, N*H_out*W_out) = (K, N*H_out*W_out)
    out = w_row @ cols + b.reshape(K, 1)

    # Reshape back to (N, K, H_out, W_out)
    out = out.reshape(K, N, H_out, W_out).transpose(1, 0, 2, 3)

    return out


# ---------------------------------------------------------------------------
# Pooling Layer
# ---------------------------------------------------------------------------

def pool_forward(x, pool_size=2, stride=2):
    """
    Max pooling forward pass.

    CS231n: "The most common form is a pooling layer with filters of size
    2x2 applied with a stride of 2 downsamples every depth slice in the
    input by 2 along both width and height, discarding 75% of the
    activations."

    Output size: (W - F) / S + 1 (no padding for pooling)

    Args:
        x: Input, shape (N, C, H, W)
        pool_size: Spatial extent of pooling window
        stride: Step size

    Returns:
        out: Pooled output, shape (N, C, H_out, W_out)
    """
    N, C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = x[:, :, h_start:h_start + pool_size, w_start:w_start + pool_size]
            out[:, :, i, j] = np.max(region, axis=(2, 3))

    return out


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

def relu(x):
    """
    ReLU activation: max(0, x).

    CS231n: "RELU layer will apply an elementwise activation function,
    such as the max(0,x) thresholding at zero."
    """
    return np.maximum(0, x)


# ---------------------------------------------------------------------------
# Fully Connected Layer
# ---------------------------------------------------------------------------

def fc_forward(x, w, b):
    """
    Fully connected (dense) layer forward pass.

    CS231n: "Each neuron in this layer will be connected to all the
    numbers in the previous volume."

    Args:
        x: Input, shape (N, D) where D = flattened volume size
        w: Weights, shape (D, M)
        b: Biases, shape (M,)

    Returns:
        out: shape (N, M)
    """
    return x @ w + b


# ---------------------------------------------------------------------------
# Softmax and Cross-Entropy Loss
# ---------------------------------------------------------------------------

def softmax(scores):
    """Numerically stable softmax."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    """
    Cross-entropy loss for classification.

    Args:
        probs: Softmax probabilities, shape (N, C)
        y: True labels, shape (N,) — integer class indices

    Returns:
        loss: Scalar average cross-entropy loss
    """
    N = probs.shape[0]
    log_probs = -np.log(probs[np.arange(N), y] + 1e-12)
    return np.mean(log_probs)


# ---------------------------------------------------------------------------
# SimpleCNN: A minimal ConvNet for CIFAR-10
# ---------------------------------------------------------------------------

class SimpleCNN:
    """
    A minimal CNN following the CS231n architecture pattern:
    INPUT -> CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FC -> RELU -> FC

    Architecture (for 32x32x3 CIFAR-10 input):
      CONV1: 8 filters, 3x3, stride 1, pad 1  -> [32x32x8]
      RELU
      POOL: 2x2, stride 2                      -> [16x16x8]
      CONV2: 16 filters, 3x3, stride 1, pad 1  -> [16x16x16]
      RELU
      POOL: 2x2, stride 2                      -> [8x8x16]
      FC1: 8*8*16=1024 -> 64
      RELU
      FC2: 64 -> 10 (num classes)

    This follows the CS231n pattern:
    INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
    """

    def __init__(self, num_classes=10, input_channels=3):
        """Initialize weights using He initialization (variance = 2/fan_in)."""
        self.params = {}

        # CONV1: 8 filters, 3x3
        fan_in = input_channels * 3 * 3
        self.params['W1'] = np.random.randn(8, input_channels, 3, 3) * np.sqrt(2.0 / fan_in)
        self.params['b1'] = np.zeros(8)

        # CONV2: 16 filters, 3x3
        fan_in = 8 * 3 * 3
        self.params['W2'] = np.random.randn(16, 8, 3, 3) * np.sqrt(2.0 / fan_in)
        self.params['b2'] = np.zeros(16)

        # FC1: 1024 -> 64
        fan_in = 8 * 8 * 16
        self.params['W3'] = np.random.randn(fan_in, 64) * np.sqrt(2.0 / fan_in)
        self.params['b3'] = np.zeros(64)

        # FC2: 64 -> num_classes
        fan_in = 64
        self.params['W4'] = np.random.randn(64, num_classes) * np.sqrt(2.0 / fan_in)
        self.params['b4'] = np.zeros(num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input images, shape (N, 3, 32, 32)

        Returns:
            scores: Class scores, shape (N, num_classes)
        """
        # CONV1 -> RELU -> POOL
        h1 = conv_forward_im2col(x, self.params['W1'], self.params['b1'], stride=1, pad=1)
        h1 = relu(h1)
        h1 = pool_forward(h1, pool_size=2, stride=2)

        # CONV2 -> RELU -> POOL
        h2 = conv_forward_im2col(h1, self.params['W2'], self.params['b2'], stride=1, pad=1)
        h2 = relu(h2)
        h2 = pool_forward(h2, pool_size=2, stride=2)

        # Flatten
        N = x.shape[0]
        h2_flat = h2.reshape(N, -1)

        # FC1 -> RELU
        h3 = fc_forward(h2_flat, self.params['W3'], self.params['b3'])
        h3 = relu(h3)

        # FC2 (scores)
        scores = fc_forward(h3, self.params['W4'], self.params['b4'])

        return scores

    def predict(self, x):
        """Return predicted class labels."""
        scores = self.forward(x)
        return np.argmax(scores, axis=1)

    def count_parameters(self):
        """Count total learnable parameters."""
        total = 0
        for name, param in self.params.items():
            total += param.size
        return total


# ---------------------------------------------------------------------------
# Utility: Parameter Counter for Arbitrary Architectures
# ---------------------------------------------------------------------------

def count_conv_params(layers):
    """
    Count parameters for a VGGNet-style architecture.

    Args:
        layers: List of tuples describing the architecture.
                Each tuple: ('conv', in_depth, num_filters, filter_size)
                        or: ('pool',)
                        or: ('fc', in_size, out_size)

    Returns:
        total_params: Total parameter count
        per_layer: List of (layer_desc, param_count) tuples
    """
    per_layer = []
    total = 0

    for layer in layers:
        if layer[0] == 'conv':
            _, d_in, k, f = layer
            # Params: (F * F * D_in + 1) * K  (weights + biases)
            params = (f * f * d_in + 1) * k
            per_layer.append((f"CONV{f}-{k}", params))
            total += params
        elif layer[0] == 'pool':
            per_layer.append(("POOL", 0))
        elif layer[0] == 'fc':
            _, d_in, d_out = layer
            params = (d_in + 1) * d_out
            per_layer.append((f"FC-{d_out}", params))
            total += params

    return total, per_layer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Run a quick demonstration of all CNN components."""
    print("=" * 60)
    print("Day 28: CNN Implementation Demo")
    print("=" * 60)

    # 1. Convolution demo
    print("\n--- Convolutional Layer ---")
    x = np.random.randn(1, 3, 8, 8)  # 1 image, 3 channels, 8x8
    w = np.random.randn(4, 3, 3, 3)  # 4 filters, 3x3
    b = np.zeros(4)

    out_naive = conv_forward_naive(x, w, b, stride=1, pad=1)
    out_im2col = conv_forward_im2col(x, w, b, stride=1, pad=1)

    print(f"Input shape:          {x.shape}")
    print(f"Filter shape:         {w.shape}")
    print(f"Output shape (naive): {out_naive.shape}")
    print(f"Output shape (im2col):{out_im2col.shape}")
    print(f"Max difference:       {np.max(np.abs(out_naive - out_im2col)):.2e}")
    print(f"  (should be ~0, confirming both methods are equivalent)")

    # Expected output size: (8 - 3 + 2*1)/1 + 1 = 8
    expected_h = (8 - 3 + 2 * 1) // 1 + 1
    print(f"Expected output H/W:  {expected_h} (matches: {out_naive.shape[2] == expected_h})")

    # 2. Pooling demo
    print("\n--- Pooling Layer ---")
    pool_in = np.random.randn(1, 4, 8, 8)
    pool_out = pool_forward(pool_in, pool_size=2, stride=2)
    print(f"Input shape:  {pool_in.shape}")
    print(f"Output shape: {pool_out.shape}")
    print(f"  (2x2 pooling with stride 2 halves spatial dimensions)")

    # 3. Full network demo
    print("\n--- SimpleCNN Forward Pass ---")
    model = SimpleCNN(num_classes=10)
    images = np.random.randn(2, 3, 32, 32)  # 2 CIFAR-10-sized images
    scores = model.forward(images)
    print(f"Input shape:  {images.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Predictions:  {model.predict(images)}")
    print(f"Total params: {model.count_parameters():,}")

    # 4. Parameter counting demo (VGGNet-16 first few layers)
    print("\n--- VGGNet-16 Parameter Count (first block) ---")
    vgg_layers = [
        ('conv', 3, 64, 3),      # CONV3-64
        ('conv', 64, 64, 3),     # CONV3-64
        ('pool',),               # POOL
        ('conv', 64, 128, 3),    # CONV3-128
        ('conv', 128, 128, 3),   # CONV3-128
        ('pool',),               # POOL
        ('fc', 7*7*512, 4096),   # FC-4096 (simplified)
    ]
    total, per_layer = count_conv_params(vgg_layers)
    for desc, count in per_layer:
        print(f"  {desc:15s} {count:>12,} params")
    print(f"  {'TOTAL':15s} {total:>12,} params")

    print("\n" + "=" * 60)
    print("All components working. Run exercises/ to practice building these.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Implementation Demo")
    parser.add_argument('--demo', action='store_true', default=True,
                        help='Run the demonstration')
    args = parser.parse_args()
    demo()
