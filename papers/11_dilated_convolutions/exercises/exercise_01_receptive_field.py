"""
Exercise 1: Understand Receptive Fields
=======================================

Goal: Visualize how dilation expands receptive fields.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def calculate_receptive_field(kernel_size, dilation, num_layers):
    """
    Calculate receptive field after stacking layers.
    
    For a single dilated conv:
        RF = (kernel_size - 1) * dilation + 1
        
    For stacked layers:
        Each layer adds (kernel_size - 1) * dilation to previous RF
    """
    # TODO 1: Calculate receptive field
    rf = 1  # Start with 1 pixel
    
    for i in range(num_layers):
        d = dilation if isinstance(dilation, int) else dilation[i]
        k = kernel_size
        
        # TODO: rf += (k - 1) * d
        pass
    
    return rf


def visualize_receptive_field():
    """
    Visualize receptive field for different dilation rates.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    dilations = [1, 2, 4, 8, 16, 32]
    kernel_size = 3
    
    for ax, d in zip(axes.flat, dilations):
        # TODO 2: Create a grid showing the receptive field pattern
        # For dilation d and kernel 3x3:
        # The actual positions sampled are at intervals of d
        
        grid = np.zeros((33, 33))
        center = 16
        
        # TODO: Mark positions that are part of the receptive field
        # for offset in [-(k//2)*d, 0, (k//2)*d]:
        #     grid[center + offset_y, center + offset_x] = 1
        
        ax.imshow(grid, cmap='Blues')
        ax.set_title(f'Dilation = {d}\nRF = {(kernel_size-1)*d + 1}x{(kernel_size-1)*d + 1}')
        ax.axis('off')
    
    plt.suptitle('Receptive Field with Different Dilation Rates', fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_rf_growth():
    """
    Compare receptive field growth: standard vs dilated stacking.
    """
    num_layers = 8
    kernel_size = 3
    
    # Standard convolutions (all dilation=1)
    standard_rf = []
    rf = 1
    for _ in range(num_layers):
        rf += (kernel_size - 1) * 1
        standard_rf.append(rf)
    
    # Dilated convolutions (dilation doubles each layer)
    dilated_rf = []
    rf = 1
    for i in range(num_layers):
        d = 2 ** i  # 1, 2, 4, 8, ...
        rf += (kernel_size - 1) * d
        dilated_rf.append(rf)
    
    # TODO 3: Plot comparison
    plt.figure(figsize=(10, 6))
    # TODO: plt.plot(range(1, num_layers+1), standard_rf, label='Standard')
    # TODO: plt.plot(range(1, num_layers+1), dilated_rf, label='Dilated')
    plt.xlabel('Number of Layers')
    plt.ylabel('Receptive Field Size')
    plt.title('Receptive Field Growth: Standard vs Dilated')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


class DilatedConv(nn.Module):
    """
    A dilated convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        
        # TODO 4: Create dilated convolution
        # Key: padding = dilation to maintain spatial size
        self.conv = None  # TODO: nn.Conv2d(in_channels, out_channels, kernel_size,
                         #                  padding=dilation, dilation=dilation)
    
    def forward(self, x):
        return self.conv(x)


def test_output_sizes():
    """
    Verify that dilated convolutions preserve spatial dimensions.
    """
    print("Testing output sizes with different dilations...")
    print("=" * 50)
    
    x = torch.randn(1, 64, 32, 32)
    print(f"Input: {list(x.shape)}")
    
    for d in [1, 2, 4, 8]:
        # TODO 5: Create dilated conv and check output size
        # conv = DilatedConv(64, 64, kernel_size=3, dilation=d)
        # y = conv(x)
        # print(f"Dilation {d}: {list(y.shape)}")
        pass


if __name__ == "__main__":
    print(__doc__)
    print("Fill in the TODOs and run the exercises!")
    
    # visualize_receptive_field()
    # compare_rf_growth()
    # test_output_sizes()
