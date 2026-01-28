"""
Exercise 1: Build ResNet-18
===========================

Goal: Implement ResNet-18 with skip connections from scratch.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    
    Structure:
        x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+x) → ReLU
        
    The key is: output = F(x) + x (skip connection)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # TODO 1: First conv layer
        self.conv1 = None  # TODO: nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = None    # TODO: nn.BatchNorm2d(out_channels)
        
        # TODO 2: Second conv layer
        self.conv2 = None  # TODO: nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = None    # TODO: nn.BatchNorm2d(out_channels)
        
        # Downsample for dimension mismatch
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        # TODO 3: Main path
        out = None  # TODO: F.relu(self.bn1(self.conv1(x)))
        out = None  # TODO: self.bn2(self.conv2(out))
        
        # TODO 4: Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # TODO 5: Add skip connection and final ReLU
        out = None  # TODO: out + identity
        out = None  # TODO: F.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 architecture.
    
    Structure:
        Conv1 (7x7, 64) → MaxPool
        Layer1: 2 BasicBlocks (64)
        Layer2: 2 BasicBlocks (128, stride 2)
        Layer3: 2 BasicBlocks (256, stride 2)
        Layer4: 2 BasicBlocks (512, stride 2)
        AvgPool → FC
    """
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.in_channels = 64
        
        # TODO 6: Initial convolution
        self.conv1 = None  # TODO: nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = None    # TODO: nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # TODO 7: Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # TODO 8: Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None  # TODO: nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a layer with num_blocks BasicBlocks."""
        downsample = None
        
        # TODO 9: Create downsample if dimensions change
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                # TODO: nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                # TODO: nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        
        # TODO 10: First block (may downsample)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        # TODO 11: Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # TODO 12: Forward pass
        x = None  # TODO: F.relu(self.bn1(self.conv1(x)))
        x = None  # TODO: self.maxpool(x)
        
        x = None  # TODO: self.layer1(x)
        x = None  # TODO: self.layer2(x)
        x = None  # TODO: self.layer3(x)
        x = None  # TODO: self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def test_resnet():
    """Test ResNet-18 implementation."""
    print("Testing ResNet-18 Implementation...")
    print("=" * 60)
    
    model = ResNet18(num_classes=1000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: ~11.7 million")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"\nInput: {list(x.shape)} → Output: {list(y.shape)}")
    print(f"Expected: [2, 3, 224, 224] → [2, 1000]")


if __name__ == "__main__":
    print(__doc__)
    # test_resnet()
