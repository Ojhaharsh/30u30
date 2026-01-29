"""
Exercise 1: Pre vs Post Activation
==================================

Goal: Compare pre-activation and post-activation residual blocks.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostActBasicBlock(nn.Module):
    """
    Original ResNet V1 block (post-activation).
    
    Structure: Conv → BN → ReLU → Conv → BN → Add → ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # TODO 1: Implement post-activation block
        self.conv1 = None  # TODO
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # TODO 2: Post-activation forward
        # out = relu(bn(conv(x)))
        # out = bn(conv(out))
        # out = relu(out + identity)
        out = None  # TODO
        
        return out


class PreActBasicBlock(nn.Module):
    """
    ResNet V2 block (pre-activation).
    
    Structure: BN → ReLU → Conv → BN → ReLU → Conv → Add
    
    Key insight: The skip connection is a pure identity!
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # TODO 3: Implement pre-activation block
        self.bn1 = None  # TODO
        self.conv1 = None
        self.bn2 = None
        self.conv2 = None
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # TODO 4: Pre-activation forward
        # out = conv(relu(bn(x)))
        # out = conv(relu(bn(out)))
        # out = out + identity  # Pure identity!
        out = None  # TODO
        
        return out


def compare_gradient_flow():
    """
    Compare gradient magnitude through both block types.
    """
    print("Comparing gradient flow...")
    
    # TODO 5: Create both block types
    post_block = None  # TODO
    pre_block = None   # TODO
    
    # Create input requiring grad
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    
    # Forward through post-activation
    # ...
    
    # Forward through pre-activation
    # ...
    
    # TODO 6: Compare gradient magnitudes
    # print("Post-activation gradient magnitude: ...")
    # print("Pre-activation gradient magnitude: ...")


if __name__ == "__main__":
    print(__doc__)
    # compare_gradient_flow()
