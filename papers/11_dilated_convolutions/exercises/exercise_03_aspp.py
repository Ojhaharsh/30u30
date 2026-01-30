"""
Exercise 3: Implement ASPP
==========================

Goal: Build Atrous Spatial Pyramid Pooling (DeepLab).

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Branches:
    1. 1x1 convolution
    2. 3x3 conv with dilation 6
    3. 3x3 conv with dilation 12
    4. 3x3 conv with dilation 18
    5. Global average pooling
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # TODO 1: 1x1 convolution branch
        self.conv1x1 = None  # TODO
        
        # TODO 2: Dilated convolution branches
        self.conv_d6 = None   # dilation=6
        self.conv_d12 = None  # dilation=12
        self.conv_d18 = None  # dilation=18
        
        # TODO 3: Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(in_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )
        
        # TODO 4: Fusion layer
        self.project = None  # Combine all 5 branches
        
    def forward(self, x):
        size = x.shape[2:]
        
        # TODO 5: Forward through all branches
        # out1 = self.conv1x1(x)
        # out2 = self.conv_d6(x)
        # out3 = self.conv_d12(x)
        # out4 = self.conv_d18(x)
        # out5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear')
        
        # TODO 6: Concatenate and project
        # concat = torch.cat([out1, out2, out3, out4, out5], dim=1)
        # return self.project(concat)
        pass


if __name__ == "__main__":
    print(__doc__)
