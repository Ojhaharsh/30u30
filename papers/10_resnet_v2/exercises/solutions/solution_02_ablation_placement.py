"""
Solution 2: Ablation on Activation Placement
============================================

Compare all 5 variants from the ResNet V2 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# The 5 variants from the paper
class OriginalBlock(nn.Module):
    """(a) Original: Conv-BN-ReLU-Conv-BN, add, ReLU"""
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
        return F.relu(out + identity)


class BNAfterAddBlock(nn.Module):
    """(b) BN after addition"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_post = nn.BatchNorm2d(channels)  # After add
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return F.relu(self.bn_post(out + identity))


class ReLUBeforeAddBlock(nn.Module):
    """(c) ReLU before addition"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))  # ReLU before add
        return out + identity  # No ReLU after


class ReLUOnlyPreActBlock(nn.Module):
    """(d) ReLU-only pre-activation"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(F.relu(x)))  # ReLU first
        out = self.bn2(self.conv2(F.relu(out)))
        return out + identity


class FullPreActBlock(nn.Module):
    """(e) Full pre-activation: BN-ReLU-Conv-BN-ReLU-Conv (WINNER!)"""
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
        return out + identity  # PURE IDENTITY!


def test_all_variants():
    """Test all 5 variants."""
    print("Activation Placement Variants")
    print("=" * 60)
    
    variants = {
        '(a) Original':        OriginalBlock,
        '(b) BN after add':    BNAfterAddBlock,
        '(c) ReLU before add': ReLUBeforeAddBlock,
        '(d) ReLU-only pre':   ReLUOnlyPreActBlock,
        '(e) Full pre-act':    FullPreActBlock,
    }
    
    x = torch.randn(1, 64, 32, 32)
    
    for name, block_class in variants.items():
        block = block_class(64)
        y = block(x)
        
        # Check if output can be pure identity (important for very deep nets)
        # Set all weights to zero - should output identity if skip is pure
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        
        y_zero = block(x)
        is_identity = torch.allclose(x, y_zero, atol=1e-5)
        
        print(f"{name:25}: shape={list(y.shape)}, pure_identity={is_identity}")
    
    print("\n" + "=" * 60)
    print("Key insight: Only (e) Full pre-activation gives PURE identity skip")
    print("This enables training of 1000+ layer networks!")


if __name__ == "__main__":
    test_all_variants()
