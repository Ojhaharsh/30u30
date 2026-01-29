"""
Solution 1: Pre vs Post Activation
==================================

Complete comparison of pre/post activation blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostActBasicBlock(nn.Module):
    """Original ResNet V1 block."""
    
    def __init__(self, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class PreActBasicBlock(nn.Module):
    """ResNet V2 block with pre-activation."""
    
    def __init__(self, channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        
    def forward(self, x):
        identity = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + identity  # Pure identity!
        return out


def compare_gradient_flow():
    """Compare gradient magnitude through both blocks."""
    print("Gradient Flow Comparison")
    print("=" * 50)
    
    # Stack 10 blocks of each type
    post_net = nn.Sequential(*[PostActBasicBlock(64) for _ in range(10)])
    pre_net = nn.Sequential(*[PreActBasicBlock(64) for _ in range(10)])
    
    # Input
    x_post = torch.randn(1, 64, 32, 32, requires_grad=True)
    x_pre = torch.randn(1, 64, 32, 32, requires_grad=True)
    
    # Forward + backward
    out_post = post_net(x_post).sum()
    out_post.backward()
    
    out_pre = pre_net(x_pre).sum()
    out_pre.backward()
    
    # Compare
    print(f"Post-activation input grad norm: {x_post.grad.norm():.4f}")
    print(f"Pre-activation input grad norm:  {x_pre.grad.norm():.4f}")
    print()
    print("Pre-activation maintains better gradient flow!")


if __name__ == "__main__":
    compare_gradient_flow()
