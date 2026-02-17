"""
exercise_05_feature_viz.py - Visualize CNN Features

Difficulty: Hard (4/5)

Task: Implement two visualization techniques to understand what a CNN learns:
1. Filter visualization — display the learned weights as small images
2. Activation maximization — generate an input that maximally activates
   a given filter (gradient ascent in input space)

The goal is to see that early layers learn edge detectors, while deeper
layers learn more complex patterns.

Reference: CS231n - https://cs231n.github.io/understanding-cnn/
"""

import numpy as np
import sys
import os

# Add parent directory to path so we can import implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import SimpleCNN, conv_forward_im2col, relu


def visualize_filters(filters, save_path=None):
    """
    Visualize convolutional filters as small images.

    YOUR TASK: Fill in this function.

    For first-layer filters (operating on RGB), each 3xFHxFW filter can be
    displayed as an FH x FW RGB image after normalization to [0, 1].

    Args:
        filters: shape (K, C, FH, FW) — K filters
        save_path: If provided, save the plot

    Hints:
        1. Import matplotlib.pyplot
        2. Create a grid of subplots (e.g., 4 rows x K//4 cols)
        3. For each filter, transpose to (FH, FW, C) for imshow
        4. Normalize each filter to [0, 1]: (f - f.min()) / (f.max() - f.min())
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[NOTE] matplotlib not installed. Skipping visualization.")
        return

    raise NotImplementedError("Implement filter visualization")

    # K = filters.shape[0]
    # cols = min(8, K)
    # rows = (K + cols - 1) // cols
    #
    # fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    # fig.suptitle("Learned Filters", fontweight='bold')
    #
    # for idx in range(K):
    #     ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx]
    #     f = filters[idx].transpose(1, 2, 0)  # CHW -> HWC
    #     f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    #     ax.imshow(f)
    #     ax.axis('off')
    #
    # plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=150)
    # plt.show()


def compute_activation_map(image, filters, biases, layer_idx=0):
    """
    Compute activation maps for a given image and filter set.

    YOUR TASK: Fill in this function.

    Feed the image through convolution + ReLU and return the activation maps.

    Args:
        image: shape (1, C, H, W) — single image
        filters: shape (K, C, FH, FW)
        biases: shape (K,)
        layer_idx: Which layer's activations to compute (0 = first conv)

    Returns:
        activations: shape (K, H_out, W_out)

    Hints:
        1. Use conv_forward_im2col from implementation.py
        2. Apply relu
        3. Return activations[0] (remove batch dimension)
    """
    raise NotImplementedError("Implement activation map computation")

    # conv_out = conv_forward_im2col(image, filters, biases, stride=1, pad=1)
    # activations = relu(conv_out)
    # return activations[0]  # Remove batch dimension


def plot_activation_maps(activations, save_path=None):
    """
    Display activation maps in a grid.

    YOUR TASK: Fill in this function.

    Args:
        activations: shape (K, H, W) — K activation maps
        save_path: If provided, save the plot

    Hints:
        1. Create subplots grid
        2. Use imshow with cmap='viridis' for each map
        3. Title each map with its index
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[NOTE] matplotlib not installed. Skipping visualization.")
        return

    raise NotImplementedError("Implement activation map plotting")

    # K = activations.shape[0]
    # cols = min(8, K)
    # rows = (K + cols - 1) // cols
    #
    # fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    # fig.suptitle("Activation Maps", fontweight='bold')
    #
    # for idx in range(K):
    #     ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx]
    #     ax.imshow(activations[idx], cmap='viridis')
    #     ax.set_title(f"Map {idx}", fontsize=8)
    #     ax.axis('off')
    #
    # plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=150)
    # plt.show()


def check():
    """Test your implementation."""
    np.random.seed(42)

    # Test 1: Filter visualization
    print("Test 1: Filter visualization")
    model = SimpleCNN()
    filters = model.params['W1']
    print(f"  First-layer filters shape: {filters.shape}")
    print(f"  (K={filters.shape[0]} filters, C={filters.shape[1]} channels, "
          f"{filters.shape[2]}x{filters.shape[3]} spatial)")
    try:
        visualize_filters(filters, save_path="test_filters.png")
        print("[OK] Filter visualization generated")
        if os.path.exists("test_filters.png"):
            os.remove("test_filters.png")
    except NotImplementedError:
        print("[--] Not implemented yet")

    # Test 2: Activation maps
    print("\nTest 2: Activation maps")
    image = np.random.randn(1, 3, 32, 32)
    try:
        activations = compute_activation_map(
            image, model.params['W1'], model.params['b1']
        )
        assert activations.shape == (8, 32, 32), \
            f"Expected (8, 32, 32), got {activations.shape}"
        print(f"[OK] Activation maps shape: {activations.shape}")
        # Check ReLU was applied
        assert np.all(activations >= 0), "Activations should be non-negative (ReLU)"
        print("[OK] All activations >= 0 (ReLU applied)")
    except NotImplementedError:
        print("[--] Not implemented yet")

    # Test 3: Plot activation maps
    print("\nTest 3: Plot activation maps")
    try:
        activations = compute_activation_map(
            image, model.params['W1'], model.params['b1']
        )
        plot_activation_maps(activations, save_path="test_activations.png")
        print("[OK] Activation map plot generated")
        if os.path.exists("test_activations.png"):
            os.remove("test_activations.png")
    except NotImplementedError:
        print("[--] Not implemented yet")

    print("\nAll implemented tests passed.")


if __name__ == "__main__":
    check()
