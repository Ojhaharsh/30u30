"""
Solution 1: Receptive Field Understanding
=========================================

Complete solution for receptive field visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def calculate_receptive_field(kernel_size, dilations):
    """Calculate total receptive field."""
    rf = 1
    for d in dilations:
        rf += (kernel_size - 1) * d
    return rf


def visualize_dilated_kernel():
    """Visualize how dilation spreads the kernel."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    dilations = [1, 2, 4, 8, 16, 32]
    kernel_size = 3
    
    for ax, d in zip(axes.flat, dilations):
        # Calculate positions for 3x3 kernel with dilation d
        size = (kernel_size - 1) * d + 1
        grid = np.zeros((size, size))
        
        # Mark kernel positions
        for i in range(kernel_size):
            for j in range(kernel_size):
                grid[i * d, j * d] = 1
        
        ax.imshow(grid, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Dilation = {d}\nRF = {size}×{size}')
        ax.axis('off')
        
        # Mark center
        center = (kernel_size // 2) * d
        ax.plot(center, center, 'ro', markersize=10)
    
    plt.suptitle('3×3 Kernel with Different Dilations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_rf_growth():
    """Compare RF growth: standard vs exponentially dilated."""
    num_layers = 10
    k = 3
    
    # Standard: all dilation=1
    standard = [1]
    for _ in range(num_layers):
        standard.append(standard[-1] + (k-1) * 1)
    
    # Exponential dilation: 1, 2, 4, 8, ...
    exponential = [1]
    for i in range(num_layers):
        d = 2 ** i
        exponential.append(exponential[-1] + (k-1) * d)
    
    # Linear dilation: 1, 2, 3, 4, ...
    linear = [1]
    for i in range(num_layers):
        d = i + 1
        linear.append(linear[-1] + (k-1) * d)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(num_layers + 1), standard, 'b-o', label='Standard (d=1)', linewidth=2)
    plt.semilogy(range(num_layers + 1), linear, 'g-s', label='Linear (d=1,2,3...)', linewidth=2)
    plt.semilogy(range(num_layers + 1), exponential, 'r-^', label='Exponential (d=1,2,4...)', linewidth=2)
    
    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('Receptive Field Size (log scale)', fontsize=12)
    plt.title('Receptive Field Growth Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"After {num_layers} layers:")
    print(f"  Standard:     RF = {standard[-1]}")
    print(f"  Linear:       RF = {linear[-1]}")
    print(f"  Exponential:  RF = {exponential[-1]}")


def test_dilated_conv():
    """Test dilated convolution preserves spatial size."""
    print("Dilated Convolution Output Sizes")
    print("=" * 50)
    
    x = torch.randn(1, 64, 128, 128)
    print(f"Input: {list(x.shape)}")
    print()
    
    for d in [1, 2, 4, 8, 16]:
        # padding = dilation preserves size
        conv = nn.Conv2d(64, 64, kernel_size=3, padding=d, dilation=d)
        y = conv(x)
        print(f"Dilation {d:2d}: {list(y.shape)} (RF contribution: {(3-1)*d + 1})")
    
    print()
    print("✅ All outputs have the same spatial size!")


if __name__ == "__main__":
    print("Receptive Field Visualization")
    print("=" * 50)
    
    visualize_dilated_kernel()
    compare_rf_growth()
    test_dilated_conv()
