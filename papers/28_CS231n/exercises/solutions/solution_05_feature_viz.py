"""
solution_05_feature_viz.py - Solution for Exercise 5

Visualizes CNN filters and activation maps.

Reference: CS231n - https://cs231n.github.io/understanding-cnn/
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import SimpleCNN, conv_forward_im2col, relu


def visualize_filters(filters, save_path=None):
    """
    Visualize convolutional filters as small images.

    For first-layer filters (RGB input), each filter can be displayed
    as an FH x FW color image after normalization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[NOTE] matplotlib not installed. Skipping visualization.")
        return

    K = filters.shape[0]
    cols = min(8, K)
    rows = (K + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle("Learned Filters", fontweight='bold')

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < K:
                f = filters[idx]
                if f.shape[0] == 3:  # RGB
                    f_display = f.transpose(1, 2, 0)  # CHW -> HWC for imshow
                else:
                    f_display = f[0]  # Just first channel
                # Normalize to [0, 1]
                f_display = (f_display - f_display.min()) / (f_display.max() - f_display.min() + 1e-8)
                if len(f_display.shape) == 2:
                    ax.imshow(f_display, cmap='gray')
                else:
                    ax.imshow(f_display)
                ax.set_title(f"F{idx}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def compute_activation_map(image, filters, biases, layer_idx=0):
    """
    Compute activation maps by passing image through conv + ReLU.

    Returns shape (K, H_out, W_out) — one map per filter.
    """
    conv_out = conv_forward_im2col(image, filters, biases, stride=1, pad=1)
    activations = relu(conv_out)
    return activations[0]  # Remove batch dimension -> (K, H, W)


def plot_activation_maps(activations, save_path=None):
    """
    Display activation maps in a grid.

    Each map shows where a particular filter "fires" spatially.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[NOTE] matplotlib not installed. Skipping visualization.")
        return

    K = activations.shape[0]
    cols = min(8, K)
    rows = (K + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle("Activation Maps", fontweight='bold')

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < K:
                ax.imshow(activations[idx], cmap='viridis')
                ax.set_title(f"Map {idx}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def check():
    """Test the implementation."""
    np.random.seed(42)

    # Test 1: Filter visualization
    print("Test 1: Filter visualization")
    model = SimpleCNN()
    filters = model.params['W1']
    print(f"  First-layer filters shape: {filters.shape}")
    visualize_filters(filters, save_path="test_filters.png")
    print("[OK] Filter visualization generated")
    if os.path.exists("test_filters.png"):
        os.remove("test_filters.png")

    # Test 2: Activation maps
    print("\nTest 2: Activation maps")
    image = np.random.randn(1, 3, 32, 32)
    activations = compute_activation_map(image, model.params['W1'], model.params['b1'])
    assert activations.shape == (8, 32, 32), f"Expected (8,32,32), got {activations.shape}"
    print(f"[OK] Activation maps shape: {activations.shape}")
    assert np.all(activations >= 0), "Activations should be non-negative (ReLU)"
    print("[OK] All activations >= 0 (ReLU applied)")

    # Test 3: Plot activation maps
    print("\nTest 3: Plot activation maps")
    plot_activation_maps(activations, save_path="test_activations.png")
    print("[OK] Activation map plot generated")
    if os.path.exists("test_activations.png"):
        os.remove("test_activations.png")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
