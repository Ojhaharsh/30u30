"""
Solution 5: Transfer Learning V1 vs V2 Comparison
==================================================

Compare transfer learning performance between ResNet V1 and V2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PreActBlock(nn.Module):
    """Pre-activation block for V2."""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + identity


class ResNetV2(nn.Module):
    """ResNet V2 with pre-activation."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        
        self.bn_final = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_ch, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_ch:
            downsample = nn.Conv2d(self.in_channels, out_ch, 1, stride, bias=False)
        
        layers = [PreActBlock(self.in_channels, out_ch, stride, downsample)]
        self.in_channels = out_ch
        for _ in range(1, num_blocks):
            layers.append(PreActBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def get_resnet_v1(num_classes=10, pretrained=True):
    """Get ResNet V1 (standard PyTorch)."""
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model


def get_resnet_v2(num_classes=10):
    """Get ResNet V2 (pre-activation)."""
    return ResNetV2(num_classes=num_classes)


def compare_architectures():
    """Compare V1 vs V2 architecture."""
    print("ResNet V1 vs V2 Comparison")
    print("=" * 50)
    
    v1 = get_resnet_v1(10, pretrained=False)
    v2 = get_resnet_v2(10)
    
    v1_params = sum(p.numel() for p in v1.parameters())
    v2_params = sum(p.numel() for p in v2.parameters())
    
    print(f"V1 (post-activation): {v1_params:,} params")
    print(f"V2 (pre-activation):  {v2_params:,} params")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y1 = v1(x)
    y2 = v2(x)
    
    print(f"\nForward pass: {list(x.shape)} → {list(y1.shape)}")
    print("\nKey differences:")
    print("  V1: Conv → BN → ReLU → Conv → BN → Add → ReLU")
    print("  V2: BN → ReLU → Conv → BN → ReLU → Conv → Add")
    print("\nV2 advantages:")
    print("  - Pure identity skip connection")
    print("  - Better gradient flow for very deep nets")
    print("  - Slightly faster convergence")


if __name__ == "__main__":
    compare_architectures()
