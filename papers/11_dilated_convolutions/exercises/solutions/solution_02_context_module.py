"""
Solution 2: Context Module
==========================

Multi-scale context aggregation with parallel dilated convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DilatedContextModule(nn.Module):
    """
    Multi-scale context module using parallel dilated convolutions.
    
    Each branch captures context at a different scale.
    Outputs are concatenated and fused.
    """
    
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super().__init__()
        
        # Parallel dilated convolution branches
        self.branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Fusion: concatenate all branches and reduce
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dilations = dilations
        
    def forward(self, x):
        # Process all branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        concat = torch.cat(branch_outputs, dim=1)
        
        # Fuse to output channels
        return self.fusion(concat)


class MultiScaleNet(nn.Module):
    """Simple network with context module for segmentation."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Multi-scale context
        self.context = DilatedContextModule(64, 64, dilations=[1, 2, 4, 8])
        
        # Classifier
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.context(x)
        return self.classifier(x)


def visualize_receptive_fields():
    """Visualize receptive fields of different dilation rates."""
    dilations = [1, 2, 4, 8]
    k = 3  # kernel size
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax, d in zip(axes, dilations):
        # Create grid showing which pixels are sampled
        rf_size = (k - 1) * d + 1
        grid = torch.zeros(rf_size, rf_size)
        
        for i in range(k):
            for j in range(k):
                grid[i * d, j * d] = 1
        
        ax.imshow(grid, cmap='Blues')
        ax.set_title(f'Dilation={d}\nRF={rf_size}×{rf_size}')
        ax.axis('off')
        
        # Mark center
        center = rf_size // 2
        ax.plot(center, center, 'ro', markersize=10)
    
    plt.suptitle('Multi-Scale Context: Parallel Dilated Convolutions', fontsize=14)
    plt.tight_layout()
    plt.show()


def test_context_module():
    """Test the context module."""
    print("Multi-Scale Context Module")
    print("=" * 60)
    
    # Test basic functionality
    module = DilatedContextModule(64, 128, dilations=[1, 2, 4, 8])
    x = torch.randn(1, 64, 32, 32)
    y = module(x)
    
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(y.shape)}")
    print(f"Dilations: {module.dilations}")
    print(f"Parameters: {sum(p.numel() for p in module.parameters()):,}")
    
    # Test in full network
    print("\nFull network test:")
    net = MultiScaleNet(num_classes=21)
    x = torch.randn(1, 3, 128, 128)
    y = net(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(y.shape)} (per-pixel scores for 21 classes)")
    
    visualize_receptive_fields()


if __name__ == "__main__":
    test_context_module()
