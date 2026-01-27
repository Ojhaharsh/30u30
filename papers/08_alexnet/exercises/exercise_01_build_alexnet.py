"""
Exercise 1: Build AlexNet from Scratch
======================================

Goal: Implement the complete AlexNet architecture in PyTorch.

Your Task:
- Fill in the TODOs to build each layer
- Verify shapes match the paper
- Test forward pass

Learning Objectives:
1. Understand convolutional layer parameters
2. Calculate output dimensions
3. Connect conv layers to fully connected
4. Apply ReLU and dropout

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet Architecture (2012)
    
    Input: 224 x 224 x 3 (RGB image)
    
    Conv1: 96 filters, 11x11, stride 4, pad 2 → 55x55x96
    Pool1: 3x3 max pool, stride 2 → 27x27x96
    
    Conv2: 256 filters, 5x5, pad 2 → 27x27x256
    Pool2: 3x3 max pool, stride 2 → 13x13x256
    
    Conv3: 384 filters, 3x3, pad 1 → 13x13x384
    Conv4: 384 filters, 3x3, pad 1 → 13x13x384
    Conv5: 256 filters, 3x3, pad 1 → 13x13x256
    Pool5: 3x3 max pool, stride 2 → 6x6x256
    
    Flatten: 6*6*256 = 9216
    FC6: 4096 (+ dropout 0.5)
    FC7: 4096 (+ dropout 0.5)
    FC8: 1000 (num_classes)
    """
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # TODO 1: Conv1 - 96 filters, 11x11, stride 4, padding 2
            # Input: 3 channels (RGB), Output: 96 feature maps
            # After this: 55 x 55 x 96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # TODO: Fill in
            nn.ReLU(inplace=True),
            
            # TODO 2: Pool1 - 3x3 max pooling, stride 2
            # After this: 27 x 27 x 96
            nn.MaxPool2d(kernel_size=3, stride=2),  # TODO: Fill in
            
            # TODO 3: Conv2 - 256 filters, 5x5, padding 2
            # After this: 27 x 27 x 256
            None,  # TODO: nn.Conv2d(96, 256, ...)
            nn.ReLU(inplace=True),
            
            # TODO 4: Pool2
            # After this: 13 x 13 x 256
            None,  # TODO: nn.MaxPool2d(...)
            
            # TODO 5: Conv3 - 384 filters, 3x3, padding 1
            # After this: 13 x 13 x 384
            None,  # TODO
            nn.ReLU(inplace=True),
            
            # TODO 6: Conv4 - 384 filters, 3x3, padding 1
            # After this: 13 x 13 x 384
            None,  # TODO
            nn.ReLU(inplace=True),
            
            # TODO 7: Conv5 - 256 filters, 3x3, padding 1
            # After this: 13 x 13 x 256
            None,  # TODO
            nn.ReLU(inplace=True),
            
            # TODO 8: Pool5
            # After this: 6 x 6 x 256
            None,  # TODO
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # TODO 9: Fully connected layers (classifier)
        self.classifier = nn.Sequential(
            # Dropout for regularization
            nn.Dropout(p=0.5),
            
            # FC6: 6*6*256 = 9216 → 4096
            None,  # TODO: nn.Linear(6*6*256, 4096)
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            
            # FC7: 4096 → 4096
            None,  # TODO: nn.Linear(4096, 4096)
            nn.ReLU(inplace=True),
            
            # FC8: 4096 → num_classes
            None,  # TODO: nn.Linear(4096, num_classes)
        )
        
        # TODO 10: Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights following AlexNet paper recommendations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TODO 11: Initialize conv weights
                # Paper uses: normal distribution, mean=0, std=0.01
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                pass  # TODO
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                # TODO 12: Initialize linear weights
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                pass  # TODO
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # TODO 13: Complete forward pass
        # 1. Pass through convolutional layers
        x = None  # TODO: self.features(x)
        
        # 2. Adaptive pooling
        x = self.avgpool(x)
        
        # 3. Flatten for fully connected layers
        x = None  # TODO: torch.flatten(x, 1)
        
        # 4. Pass through classifier
        x = None  # TODO: self.classifier(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_output_shapes(model):
    """Verify intermediate output shapes."""
    print("Verifying layer shapes...")
    print("=" * 60)
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Track shapes through features
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
    
    # Through avgpool
    x = model.avgpool(x)
    print(f"After avgpool: {x.shape}")
    
    # Flatten
    x = torch.flatten(x, 1)
    print(f"After flatten: {x.shape}")
    
    # Through classifier
    for i, layer in enumerate(model.classifier):
        x = layer(x)
        print(f"After classifier {i} ({layer.__class__.__name__}): {x.shape}")
    
    print("=" * 60)


def test_alexnet():
    """Test your AlexNet implementation."""
    print("Testing AlexNet Implementation...")
    print("=" * 60)
    
    # Create model
    model = AlexNet(num_classes=1000)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Expected: ~61 million")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)  # batch of 2 images
    
    try:
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Expected output: [2, 1000]")
        
        if y.shape == torch.Size([2, 1000]):
            print("\n✅ Forward pass successful!")
        else:
            print("\n❌ Output shape mismatch!")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
    
    # Verify shapes
    print("\n")
    verify_output_shapes(model)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs (search for 'TODO')")
    print("2. Run this file to test")
    print("3. Check solutions/solution_01_build_alexnet.py if stuck")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # test_alexnet()
