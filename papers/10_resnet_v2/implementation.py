"""
Day 10: Identity Mappings in Deep Residual Networks (ResNet V2)
Implementation: Pre-activation residual blocks and perfect identity mappings

This implementation demonstrates the key improvements of ResNet V2:
1. Pre-activation design (BN → ReLU → Conv)
2. Clean identity shortcut connections  
3. Better signal flow for gradient propagation
4. Enables training of extremely deep networks (1000+ layers)

Key insight: Moving batch normalization and ReLU before convolutions 
creates pristine identity paths that dramatically improve training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
import math


class PreActBlock(nn.Module):
    """
    Pre-activation residual block for ResNet V2.
    
    Key innovation: BN and ReLU come BEFORE convolutions, creating
    clean identity shortcuts for perfect gradient flow.
    
    Structure:
    x → BN → ReLU → Conv → BN → ReLU → Conv → (+) → output
    └─────────────────────────────────────────┘
           (pristine identity path)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        stride: Convolution stride (default: 1)
        expansion: Channel expansion factor (default: 1)
    """
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Pre-activation: normalization and activation BEFORE convolution
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        
        # Shortcut connection for dimension matching
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 conv for shortcut when dimensions change
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                    stride=stride, bias=False)
        else:
            self.shortcut = None
    
    def forward(self, x):
        """
        Forward pass with pre-activation design.
        
        The key insight: apply BN and ReLU to input, then pass through
        convolutions, keeping the shortcut path completely clean.
        """
        # Apply pre-activation to input
        preact = F.relu(self.bn1(x))
        
        # First convolution with pre-activated input
        out = self.conv1(preact)
        
        # Second pre-activation and convolution
        out = self.conv2(F.relu(self.bn2(out)))
        
        # Clean shortcut connection
        if self.shortcut is not None:
            # For dimension changes, apply shortcut to pre-activated input
            shortcut = self.shortcut(preact)
        else:
            # Pure identity mapping
            shortcut = x
        
        # Clean addition (no activation applied here!)
        return out + shortcut


class PreActBottleneck(nn.Module):
    """
    Pre-activation bottleneck block for deeper ResNet V2 variants.
    
    Uses 1x1 → 3x3 → 1x1 convolution pattern with channel expansion.
    All activations are pre-applied, maintaining clean identity paths.
    
    Structure:
    x → BN → ReLU → 1×1 → BN → ReLU → 3×3 → BN → ReLU → 1×1 → (+) → output
    └───────────────────────────────────────────────────────────┘
                        (pristine identity path)
    """
    
    expansion = 4  # Bottleneck expands channels by 4x
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Pre-activation batch norms
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Bottleneck convolutions: 1x1 → 3x3 → 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        
        # Shortcut for dimension matching
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * self.expansion, 
                                    kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None
    
    def forward(self, x):
        """
        Bottleneck forward pass with pre-activation.
        
        Each convolution is preceded by its own BN and ReLU,
        maintaining the pre-activation principle throughout.
        """
        # First pre-activation and 1x1 conv
        preact1 = F.relu(self.bn1(x))
        out = self.conv1(preact1)
        
        # Second pre-activation and 3x3 conv
        out = self.conv2(F.relu(self.bn2(out)))
        
        # Third pre-activation and 1x1 conv
        out = self.conv3(F.relu(self.bn3(out)))
        
        # Clean shortcut
        if self.shortcut is not None:
            shortcut = self.shortcut(preact1)
        else:
            shortcut = x
        
        return out + shortcut


class ResNetV2(nn.Module):
    """
    ResNet V2 with pre-activation blocks and perfect identity mappings.
    
    Key improvements over original ResNet:
    1. Pre-activation design (BN → ReLU → Conv)
    2. Clean identity shortcuts without any processing
    3. Better gradient flow enabling very deep networks
    4. More stable training and faster convergence
    
    Args:
        block: Block type (PreActBlock or PreActBottleneck)
        layers: Number of blocks in each layer [layer1, layer2, layer3, layer4]
        num_classes: Number of output classes (default: 1000)
        zero_init_residual: Zero-initialize last BN in each residual branch
    """
    
    def __init__(self, block, layers: List[int], num_classes: int = 1000, 
                 zero_init_residual: bool = False):
        super().__init__()
        
        self.in_channels = 64
        self.block = block
        
        # Initial convolution (no BN/ReLU - applied in first block)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with pre-activation blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final batch norm and classification
        final_channels = 512 * block.expansion
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)
        
        # Initialize weights
        self._init_weights(zero_init_residual)
    
    def _make_layer(self, block, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """
        Create a residual layer with specified number of blocks.
        
        First block may have stride > 1 for downsampling.
        Subsequent blocks have stride = 1.
        """
        layers = []
        
        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, zero_init_residual: bool):
        """
        Initialize network weights following ResNet V2 conventions.
        
        Key insight: Zero-initializing the last BN in residual branches
        helps very deep networks train from scratch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BN weights to 1, biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-init the last BN in each residual branch for better training
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, PreActBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        """
        Forward pass through ResNet V2.
        
        Note: First layer uses standard conv + maxpool since
        there's no previous layer for pre-activation.
        """
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # Residual layers with pre-activation blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final activation (since last block doesn't apply ReLU)
        x = F.relu(self.bn_final(x))
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x, layer_name: str):
        """
        Extract intermediate feature maps for analysis.
        
        Args:
            x: Input tensor
            layer_name: One of ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        
        Returns:
            Feature maps from specified layer
        """
        x = self.conv1(x)
        if layer_name == 'conv1':
            return x
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if layer_name == 'layer1':
            return x
        
        x = self.layer2(x)
        if layer_name == 'layer2':
            return x
        
        x = self.layer3(x)
        if layer_name == 'layer3':
            return x
        
        x = self.layer4(x)
        if layer_name == 'layer4':
            return x
        
        raise ValueError(f"Unknown layer: {layer_name}")


def resnet_v2_18(num_classes: int = 1000, **kwargs) -> ResNetV2:
    """ResNet V2-18 with pre-activation blocks."""
    return ResNetV2(PreActBlock, [2, 2, 2, 2], num_classes, **kwargs)


def resnet_v2_34(num_classes: int = 1000, **kwargs) -> ResNetV2:
    """ResNet V2-34 with pre-activation blocks."""
    return ResNetV2(PreActBlock, [3, 4, 6, 3], num_classes, **kwargs)


def resnet_v2_50(num_classes: int = 1000, **kwargs) -> ResNetV2:
    """ResNet V2-50 with pre-activation bottleneck blocks."""
    return ResNetV2(PreActBottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnet_v2_101(num_classes: int = 1000, **kwargs) -> ResNetV2:
    """ResNet V2-101 with pre-activation bottleneck blocks."""
    return ResNetV2(PreActBottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def resnet_v2_152(num_classes: int = 1000, **kwargs) -> ResNetV2:
    """ResNet V2-152 with pre-activation bottleneck blocks."""
    return ResNetV2(PreActBottleneck, [3, 8, 36, 3], num_classes, **kwargs)


class SignalFlowAnalyzer:
    """
    Analyze signal flow in ResNet V2 vs original ResNet.
    
    This class provides tools to study how pre-activation improves
    gradient flow and signal preservation compared to post-activation.
    """
    
    def __init__(self):
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self, model: nn.Module):
        """Register forward and backward hooks to capture signal flow."""
        
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().clone()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach().clone()
            return hook
        
        # Register hooks on key layers
        for name, module in model.named_modules():
            if 'conv' in name or 'bn' in name or 'shortcut' in name:
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def analyze_gradient_flow(self) -> Dict[str, float]:
        """
        Analyze gradient flow through the network.
        
        Returns gradient norms at different depths to assess
        gradient degradation or improvement.
        """
        gradient_norms = {}
        
        for name, grad in self.gradients.items():
            if grad is not None:
                gradient_norms[name] = grad.norm().item()
        
        return gradient_norms
    
    def measure_signal_preservation(self) -> Dict[str, float]:
        """
        Measure how well signals are preserved through skip connections.
        
        Compare input vs output statistics to assess information preservation.
        """
        preservation_scores = {}
        
        for name, activation in self.activations.items():
            if 'shortcut' in name:
                # Measure information content preservation
                mean_preservation = (activation.mean() / (activation.std() + 1e-8)).abs().mean().item()
                preservation_scores[name] = mean_preservation
        
        return preservation_scores


class PreActivationStudy:
    """
    Study the effects of pre-activation vs post-activation design.
    
    This class implements both variants and provides tools to compare
    their training dynamics, gradient flow, and performance.
    """
    
    @staticmethod
    def create_comparison_blocks(in_channels: int, out_channels: int):
        """
        Create both pre-activation and post-activation blocks for comparison.
        
        Returns:
            Tuple of (pre_activation_block, post_activation_block)
        """
        # Pre-activation block (ResNet V2 style)
        pre_act = PreActBlock(in_channels, out_channels)
        
        # Post-activation block (original ResNet style)
        class PostActBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                
                if in_ch != out_ch:
                    self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                else:
                    self.shortcut = None
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                
                shortcut = self.shortcut(x) if self.shortcut else x
                
                return F.relu(out + shortcut)  # ReLU AFTER addition
        
        post_act = PostActBlock(in_channels, out_channels)
        
        return pre_act, post_act
    
    @staticmethod
    def analyze_activation_patterns(pre_block, post_block, input_tensor):
        """
        Analyze activation patterns in pre vs post activation blocks.
        
        Returns statistics about activation sparsity and distribution.
        """
        with torch.no_grad():
            pre_output = pre_block(input_tensor)
            post_output = post_block(input_tensor)
        
        analysis = {
            'pre_activation': {
                'mean': pre_output.mean().item(),
                'std': pre_output.std().item(),
                'sparsity': (pre_output == 0).float().mean().item(),
                'max': pre_output.max().item(),
                'min': pre_output.min().item()
            },
            'post_activation': {
                'mean': post_output.mean().item(), 
                'std': post_output.std().item(),
                'sparsity': (post_output == 0).float().mean().item(),
                'max': post_output.max().item(),
                'min': post_output.min().item()
            }
        }
        
        return analysis


class VeryDeepResNetV2(nn.Module):
    """
    Extremely deep ResNet V2 (200+ layers) to demonstrate the power
    of pre-activation for training very deep networks.
    
    This would be nearly impossible to train with post-activation,
    but pre-activation makes it feasible.
    """
    
    def __init__(self, depth: int = 200, num_classes: int = 1000):
        super().__init__()
        
        assert depth >= 20, "Minimum depth is 20"
        assert (depth - 2) % 6 == 0, "Depth should be 6n+2"
        
        n = (depth - 2) // 6  # Blocks per layer
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        
        # Three main layers with increasing channels
        self.layer1 = self._make_layer(16, 16, n, stride=1, first_layer=True)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        
        self.bn_final = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, first_layer=False):
        layers = []
        
        if first_layer:
            # First layer doesn't use pre-activation on input
            layers.append(PreActBlock(in_channels, out_channels, stride))
        else:
            layers.append(PreActBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(PreActBlock(out_channels, out_channels, 1))
        
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
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Example usage and demonstrations
if __name__ == "__main__":
    # Create ResNet V2 variants
    print("Creating ResNet V2 models...")
    
    # Standard variants
    resnet18_v2 = resnet_v2_18(num_classes=10)  # For CIFAR-10
    resnet50_v2 = resnet_v2_50(num_classes=1000)  # For ImageNet
    
    print(f"ResNet V2-18 parameters: {sum(p.numel() for p in resnet18_v2.parameters()):,}")
    print(f"ResNet V2-50 parameters: {sum(p.numel() for p in resnet50_v2.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = resnet50_v2(x)
    print(f"Output shape: {output.shape}")
    
    # Very deep network demonstration
    print("\nCreating very deep ResNet V2...")
    very_deep = VeryDeepResNetV2(depth=200, num_classes=10)
    print(f"Very deep ResNet V2 (200 layers) parameters: {sum(p.numel() for p in very_deep.parameters()):,}")
    
    # Pre vs post activation comparison
    print("\nComparing pre vs post activation blocks...")
    study = PreActivationStudy()
    pre_block, post_block = study.create_comparison_blocks(64, 64)
    
    x_small = torch.randn(1, 64, 32, 32)
    analysis = study.analyze_activation_patterns(pre_block, post_block, x_small)
    
    print("Pre-activation stats:", analysis['pre_activation'])
    print("Post-activation stats:", analysis['post_activation'])
    
    print("\nResNet V2 implementation complete! Ready for perfect signal flow training.")