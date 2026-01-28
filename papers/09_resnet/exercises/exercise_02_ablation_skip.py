"""
Exercise 2: Skip Connection Ablation
====================================

Goal: Remove skip connections and observe training degradation.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainBlock(nn.Module):
    """Block WITHOUT skip connection (plain network)."""
    def __init__(self, channels):
        super().__init__()
        # TODO 1: Two conv layers without skip
        self.conv1 = None  # TODO: nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        
    def forward(self, x):
        # TODO 2: Forward WITHOUT adding identity
        out = None  # TODO: F.relu(self.bn1(self.conv1(x)))
        out = None  # TODO: F.relu(self.bn2(self.conv2(out)))
        return out


class ResidualBlock(nn.Module):
    """Block WITH skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)  # Skip connection!
        return out


def compare_training():
    """Compare plain vs residual network training."""
    # TODO 3: Create both networks and train on CIFAR-10
    # Track loss curves and observe degradation in plain network
    pass


if __name__ == "__main__":
    print(__doc__)
    # compare_training()
