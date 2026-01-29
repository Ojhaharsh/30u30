"""
Solution 4: ResNet-1001 Implementation
======================================

ResNet-1001 using pre-activation bottleneck blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


class PreActBottleneck(nn.Module):
    """Pre-activation bottleneck block."""
    expansion = 4
    
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        return out + identity


class ResNet1001(nn.Module):
    """
    ResNet-1001 for CIFAR.
    
    3 stages × 111 bottleneck blocks × 3 convs = 999 + 2 = 1001 layers
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.in_channels = 16
        
        # 3 stages with 111 blocks each
        self.layer1 = self._make_layer(16, 111, stride=1)
        self.layer2 = self._make_layer(32, 111, stride=2)
        self.layer3 = self._make_layer(64, 111, stride=2)
        
        # Final BN before classifier
        self.bn_final = nn.BatchNorm2d(64 * 4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * 4, num_classes)
        
        self._init_weights()
        
    def _make_layer(self, mid_channels, num_blocks, stride):
        out_channels = mid_channels * 4
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False)
        
        layers = [PreActBottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(PreActBottleneck(self.in_channels, mid_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Use checkpointing to save memory
        x = checkpoint_sequential(self.layer1, 10, x, use_reentrant=False)
        x = checkpoint_sequential(self.layer2, 10, x, use_reentrant=False)
        x = checkpoint_sequential(self.layer3, 10, x, use_reentrant=False)
        
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def count_layers(model):
    """Count convolutional layers."""
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))


if __name__ == "__main__":
    model = ResNet1001(num_classes=10)
    
    num_layers = count_layers(model)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"ResNet-1001")
    print(f"  Conv layers: {num_layers}")
    print(f"  Parameters: {num_params:,}")
    
    # Test forward pass (small input for CIFAR)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"  Input: {list(x.shape)} → Output: {list(y.shape)}")
