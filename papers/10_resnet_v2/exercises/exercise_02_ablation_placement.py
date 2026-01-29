"""
Exercise 2: Ablation on Activation Placement
============================================

Goal: Test all 5 placement variants from the paper.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# The 5 variants from the paper:
# (a) original
# (b) BN after addition
# (c) ReLU before addition
# (d) ReLU-only pre-activation
# (e) full pre-activation (winner!)

class OriginalBlock(nn.Module):
    """(a) Original: Conv-BN-ReLU-Conv-BN, add, ReLU"""
    def __init__(self, channels):
        super().__init__()
        # TODO 1: Implement original placement
        pass

class BNAfterAddBlock(nn.Module):
    """(b) BN after addition"""
    pass

class ReLUBeforeAddBlock(nn.Module):
    """(c) ReLU before addition"""
    pass

class ReLUOnlyPreActBlock(nn.Module):
    """(d) ReLU-only pre-activation"""
    pass

class FullPreActBlock(nn.Module):
    """(e) Full pre-activation: BN-ReLU-Conv-BN-ReLU-Conv"""
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
    def forward(self, x):
        identity = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + identity


if __name__ == "__main__":
    print(__doc__)
    print("Full pre-activation (e) should perform best!")
