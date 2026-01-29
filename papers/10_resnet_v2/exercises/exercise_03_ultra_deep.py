"""
Exercise 3: Ultra-Deep Training
===============================

Goal: Train 100+ layer networks successfully.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation block for ultra-deep networks."""
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


def create_ultra_deep_resnet(num_blocks=50, channels=64):
    """Create a very deep ResNet (num_blocks * 2 conv layers)."""
    # TODO 1: Stack many pre-activation blocks
    layers = []
    # Initial conv
    layers.append(nn.Conv2d(3, channels, 3, padding=1, bias=False))
    
    # Many blocks
    for _ in range(num_blocks):
        layers.append(PreActBlock(channels))
    
    # TODO 2: Add classification head
    
    return nn.Sequential(*layers)


if __name__ == "__main__":
    print(__doc__)
    model = create_ultra_deep_resnet(50)  # 100+ layers
    print(f"Created model with ~{50*2+1} conv layers")
