"""
visualization.py - CNN Visualization Tools

Generates visualizations to understand convolutional neural networks:
  1. First-layer filter visualization
  2. Activation maps at each layer
  3. Spatial dimension progression through the network
  4. Parameter distribution across layers (VGGNet case study)

Usage:
    python visualization.py
    python visualization.py --save-dir plots/

Reference: CS231n Course Notes - https://cs231n.github.io/understanding-cnn/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot_filters(filters, title="First-Layer Filters", save_path=None):
    """
    Visualize convolutional filters as small images.

    For the first layer, filters operate directly on RGB pixels, so they
    can be visualized as tiny images. Deeper layers operate on activation
    maps and are harder to interpret visually.

    Args:
        filters: shape (K, C, FH, FW) — K filters, C channels, FH x FW spatial
        title: Plot title
        save_path: If provided, save to this path instead of showing
    """
    K = filters.shape[0]
    cols = min(8, K)
    rows = (K + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if rows == 1:
        axes = [axes] if cols == 1 else axes
        axes = np.array(axes).reshape(1, -1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < K:
                f = filters[idx]
                # Normalize each filter to [0, 1] for display
                if f.shape[0] == 3:  # RGB
                    f_display = f.transpose(1, 2, 0)  # HWC
                    f_display = (f_display - f_display.min()) / (f_display.max() - f_display.min() + 1e-8)
                    ax.imshow(f_display)
                else:
                    # Grayscale: show first channel
                    f_display = f[0]
                    f_display = (f_display - f_display.min()) / (f_display.max() - f_display.min() + 1e-8)
                    ax.imshow(f_display, cmap='gray')
                ax.set_title(f"F{idx}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_activation_maps(activations, title="Activation Maps", save_path=None):
    """
    Visualize activation maps from a convolutional layer.

    CS231n calls these "activation maps" — each one shows where a particular
    filter "fires" across the spatial extent of the input.

    Args:
        activations: shape (C, H, W) — one image's activations from a CONV layer
        title: Plot title
        save_path: If provided, save instead of showing
    """
    C = activations.shape[0]
    cols = min(8, C)
    rows = (C + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < C:
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


def plot_spatial_progression(architecture, input_size=224, title=None, save_path=None):
    """
    Show how spatial dimensions shrink through a CNN architecture.

    This visualizes the key insight from CS231n's VGGNet case study:
    spatial dimensions decrease while depth increases.

    Args:
        architecture: List of tuples:
            ('conv', filter_size, stride, pad, num_filters)
            ('pool', pool_size, stride)
        input_size: Starting spatial dimension (assumes square input)
        title: Plot title
        save_path: If provided, save instead of showing
    """
    sizes = [input_size]
    depths = [3]  # Start with RGB
    labels = ["Input"]

    current_size = input_size
    current_depth = 3

    for layer in architecture:
        if layer[0] == 'conv':
            _, f, s, p, k = layer
            current_size = (current_size - f + 2 * p) // s + 1
            current_depth = k
            labels.append(f"Conv{f}-{k}")
        elif layer[0] == 'pool':
            _, f, s = layer
            current_size = (current_size - f) // s + 1
            labels.append(f"Pool{f}")

        sizes.append(current_size)
        depths.append(current_depth)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if title is None:
        title = "Spatial Dimension Progression Through CNN"

    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Spatial size
    x_pos = range(len(sizes))
    ax1.bar(x_pos, sizes, color='steelblue', alpha=0.7)
    ax1.set_ylabel("Spatial Size (H = W)")
    ax1.set_xlabel("Layer")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_title("Spatial Dimensions Shrink")
    for i, v in enumerate(sizes):
        ax1.text(i, v + 1, str(v), ha='center', fontsize=7)

    # Depth
    ax2.bar(x_pos, depths, color='coral', alpha=0.7)
    ax2.set_ylabel("Depth (Number of Channels)")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_title("Depth Increases")
    for i, v in enumerate(depths):
        ax2.text(i, v + 1, str(v), ha='center', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_parameter_distribution(save_path=None):
    """
    Visualize where parameters live in VGGNet-16.

    CS231n's case study shows that most parameters are in the FC layers:
    the first FC layer alone has 102M of the 138M total parameters.
    Meanwhile, most memory is consumed by the early CONV layers.
    """
    # VGGNet-16 parameter counts (from CS231n)
    layers = [
        "CONV3-64", "CONV3-64",
        "CONV3-128", "CONV3-128",
        "CONV3-256", "CONV3-256", "CONV3-256",
        "CONV3-512", "CONV3-512", "CONV3-512",
        "CONV3-512", "CONV3-512", "CONV3-512",
        "FC-4096", "FC-4096", "FC-1000"
    ]

    params = [
        1728, 36864,
        73728, 147456,
        294912, 589824, 589824,
        1179648, 2359296, 2359296,
        2359296, 2359296, 2359296,
        102760448, 16777216, 4096000
    ]

    # Memory per layer (activations, in thousands)
    memory_k = [
        3211, 3211,
        1606, 1606,
        803, 803, 803,
        401, 401, 401,
        100, 100, 100,
        4, 4, 1
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("VGGNet-16: Where Do Resources Go? (from CS231n)", fontsize=14, fontweight='bold')

    # Parameters
    colors = ['steelblue'] * 13 + ['coral'] * 3
    ax1.barh(range(len(layers)), [p / 1e6 for p in params], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers, fontsize=9)
    ax1.set_xlabel("Parameters (Millions)")
    ax1.set_title("Parameters: FC layers dominate")
    ax1.invert_yaxis()

    # Memory
    ax2.barh(range(len(layers)), [m / 1000 for m in memory_k], color='seagreen', alpha=0.8)
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels(layers, fontsize=9)
    ax2.set_xlabel("Memory per Image (millions of activations)")
    ax2.set_title("Memory: Early CONV layers dominate")
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_all(save_dir="plots"):
    """Generate all visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Day 28: CNN Visualization Suite")
    print("=" * 60)

    # 1. Random filters (as if from first CONV layer)
    print("\n1. First-layer filter visualization")
    filters = np.random.randn(16, 3, 3, 3) * 0.1
    # Simulate some learned patterns
    filters[0, :, :, :] = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]*3) * 0.5  # vertical edge
    filters[1, :, :, :] = np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]*3) * 0.5  # horizontal edge
    filters[2, :, :, :] = np.array([[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]]*3) * 0.5   # diagonal
    plot_filters(filters, title="First-Layer Filters (simulated)",
                 save_path=os.path.join(save_dir, "filters_first_layer.png"))

    # 2. Activation maps (simulated)
    print("\n2. Activation maps")
    np.random.seed(42)
    activations = np.random.randn(8, 16, 16) * 0.5
    # Make some maps look like edge responses
    for i in range(8):
        activations[i] += np.random.randn(16, 16) * 0.3
    activations = np.maximum(0, activations)  # ReLU
    plot_activation_maps(activations, title="Activation Maps After CONV1 (simulated)",
                         save_path=os.path.join(save_dir, "activation_maps.png"))

    # 3. Spatial progression (VGG-style)
    print("\n3. Spatial dimension progression")
    vgg_arch = [
        ('conv', 3, 1, 1, 64), ('conv', 3, 1, 1, 64), ('pool', 2, 2),
        ('conv', 3, 1, 1, 128), ('conv', 3, 1, 1, 128), ('pool', 2, 2),
        ('conv', 3, 1, 1, 256), ('conv', 3, 1, 1, 256), ('conv', 3, 1, 1, 256), ('pool', 2, 2),
        ('conv', 3, 1, 1, 512), ('conv', 3, 1, 1, 512), ('conv', 3, 1, 1, 512), ('pool', 2, 2),
        ('conv', 3, 1, 1, 512), ('conv', 3, 1, 1, 512), ('conv', 3, 1, 1, 512), ('pool', 2, 2),
    ]
    plot_spatial_progression(vgg_arch, input_size=224,
                             title="VGGNet-16: Spatial Dims vs Depth (from CS231n)",
                             save_path=os.path.join(save_dir, "spatial_progression.png"))

    # 4. Parameter distribution
    print("\n4. Parameter distribution (VGGNet-16)")
    plot_parameter_distribution(save_path=os.path.join(save_dir, "vgg_parameter_distribution.png"))

    print("\n" + "=" * 60)
    print(f"All plots saved to {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Visualization Tools (Day 28)")
    parser.add_argument('--save-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots/)')
    args = parser.parse_args()
    generate_all(save_dir=args.save_dir)
