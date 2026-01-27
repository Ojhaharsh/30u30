"""
Solution 1: Build AlexNet from Scratch
======================================

Complete implementation of AlexNet architecture.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """Complete AlexNet implementation."""
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1: 96 filters, 11x11, stride 4, padding 2
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 256 filters, 5x5, padding 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 384 filters, 3x3, padding 1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 384 filters, 3x3, padding 1
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 256 filters, 3x3, padding 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_shapes(model):
    print("Layer shapes with 224x224 input:")
    print("=" * 50)
    
    x = torch.randn(1, 3, 224, 224)
    
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"features[{i}] ({layer.__class__.__name__:12}): {list(x.shape)}")
    
    x = model.avgpool(x)
    print(f"avgpool: {list(x.shape)}")
    
    x = torch.flatten(x, 1)
    print(f"flatten: {list(x.shape)}")
    
    for i, layer in enumerate(model.classifier):
        x = layer(x)
        print(f"classifier[{i}] ({layer.__class__.__name__:12}): {list(x.shape)}")


if __name__ == "__main__":
    model = AlexNet(num_classes=1000)
    
    print(f"Total parameters: {count_parameters(model):,}")
    print()
    
    verify_shapes(model)
    
    print()
    print("Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input: {list(x.shape)} → Output: {list(y.shape)}")
    print("✅ Success!")
