"""
Solution 3: ASPP (Atrous Spatial Pyramid Pooling)
=================================================

DeepLab's multi-scale feature extraction module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling from DeepLabV3.
    
    Five parallel branches:
    1. 1×1 convolution
    2-4. 3×3 convolutions with dilations 6, 12, 18
    5. Global average pooling (image-level features)
    """
    
    def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18]):
        super().__init__()
        
        modules = []
        
        # 1×1 convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolution branches
        for d in dilations:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.branches = nn.ModuleList(modules)
        
        # Projection layer to combine all branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        
        outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(x)
            
            # Upsample global pooling branch to match spatial size
            if i == len(self.branches) - 1:
                out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
            
            outputs.append(out)
        
        # Concatenate all branches
        concat = torch.cat(outputs, dim=1)
        
        # Project to output channels
        return self.project(concat)


class DeepLabHead(nn.Module):
    """DeepLab segmentation head with ASPP."""
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.aspp = ASPP(in_channels, 256)
        self.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        x = self.aspp(x)
        x = F.relu(self.bn(self.conv(x)))
        return self.classifier(x)


def visualize_aspp_branches():
    """Visualize ASPP branch receptive fields."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    branches = [('1×1 conv', 1), ('d=6', 13), ('d=12', 25), ('d=18', 37), ('Global Pool', 'all')]
    
    for ax, (name, rf) in zip(axes, branches):
        if rf == 'all':
            # Global pooling sees entire image
            img = torch.ones(20, 20)
            ax.imshow(img, cmap='Reds', vmin=0, vmax=1)
            ax.set_title(f'{name}\nRF=entire image')
        else:
            # Create RF visualization
            center = 10
            img = torch.zeros(21, 21)
            
            if rf == 1:
                img[center, center] = 1
            else:
                # For dilated conv with k=3
                d = (rf - 1) // 2
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        y, x = center + i*d//2, center + j*d//2
                        if 0 <= y < 21 and 0 <= x < 21:
                            img[y, x] = 1
            
            ax.imshow(img, cmap='Blues')
            ax.set_title(f'{name}\nRF={rf}×{rf}')
        
        ax.axis('off')
    
    plt.suptitle('ASPP: Multi-Scale Feature Extraction', fontsize=14)
    plt.tight_layout()
    plt.show()


def test_aspp():
    """Test ASPP module."""
    print("ASPP (Atrous Spatial Pyramid Pooling)")
    print("=" * 60)
    
    aspp = ASPP(2048, 256)  # Typical for ResNet backbone
    x = torch.randn(1, 2048, 32, 32)
    y = aspp(x)
    
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(y.shape)}")
    print(f"Parameters: {sum(p.numel() for p in aspp.parameters()):,}")
    
    print("\nASPP branches:")
    print("  1. 1×1 conv (local)")
    print("  2. 3×3 conv, dilation=6 (medium)")
    print("  3. 3×3 conv, dilation=12 (large)")
    print("  4. 3×3 conv, dilation=18 (very large)")
    print("  5. Global avg pool (entire image)")
    
    # Test full head
    print("\nFull DeepLab head test:")
    head = DeepLabHead(512, 21)
    x = torch.randn(1, 512, 64, 64)
    y = head(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(y.shape)} (21 class scores per pixel)")
    
    visualize_aspp_branches()


if __name__ == "__main__":
    test_aspp()
