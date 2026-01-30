"""
Day 11: Multi-Scale Context Aggregation with Dilated Convolutions
Implementation: Complete dilated convolution architectures and analysis tools

This implementation demonstrates:
1. Dilated convolution operations and receptive field analysis
2. DeepLab-style Atrous Spatial Pyramid Pooling (ASPP)
3. WaveNet-inspired dilated architectures for sequential data
4. Multi-scale feature extraction without resolution loss
5. Efficient context aggregation for dense prediction tasks

Key insight: Dilated convolutions expand receptive fields exponentially
while maintaining parameter efficiency and spatial resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union
import warnings


class DilatedConv2d(nn.Module):
    """
    Dilated (Atrous) Convolution with comprehensive analysis capabilities.
    
    Expands receptive field by inserting 'holes' between kernel elements,
    allowing capture of multi-scale context without parameter increase.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Size of convolution kernel
        dilation: Dilation rate (spacing between kernel elements)
        stride: Convolution stride
        padding: Padding strategy ('same' for maintained size, or int)
        bias: Whether to use bias term
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, stride: int = 1, padding: Union[str, int] = 'same', 
                 bias: bool = False):
        super().__init__()
        
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Calculate effective kernel size with dilation
        self.effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        
        # Handle padding
        if padding == 'same':
            # Calculate padding to maintain input size
            self.padding = (self.effective_kernel_size - 1) // 2
        else:
            self.padding = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=self.padding, 
                             dilation=dilation, bias=bias)
    
    def forward(self, x):
        return self.conv(x)
    
    def get_receptive_field_size(self) -> int:
        """Calculate the receptive field size of this dilated convolution."""
        return self.effective_kernel_size
    
    def get_parameter_count(self) -> int:
        """Get number of parameters (same as standard conv regardless of dilation)."""
        return sum(p.numel() for p in self.conv.parameters())


class MultiScaleDilatedBlock(nn.Module):
    """
    Multi-scale dilated convolution block that processes multiple dilation rates.
    
    Captures features at different spatial scales simultaneously without
    losing resolution, enabling rich multi-scale representations.
    
    Architecture:
    Input → [BN → ReLU → DilatedConv(d=1,2,4,8)] → Concatenate → Fusion → Output
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 dilations: List[int] = [1, 2, 4, 8]):
        super().__init__()
        
        self.dilations = dilations
        self.num_scales = len(dilations)
        
        # Each dilation rate gets equal channel allocation
        channels_per_scale = out_channels // self.num_scales
        remaining_channels = out_channels % self.num_scales
        
        # Pre-activation for better signal flow
        self.bn = nn.BatchNorm2d(in_channels)
        
        # Parallel dilated convolutions for each scale
        self.dilated_convs = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            # Last scale gets any remaining channels
            scale_channels = channels_per_scale
            if i == len(dilations) - 1:
                scale_channels += remaining_channels
                
            conv = DilatedConv2d(in_channels, scale_channels, 
                               kernel_size=3, dilation=dilation)
            self.dilated_convs.append(conv)
        
        # Feature fusion after concatenation
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # Pre-activation
        x_activated = F.relu(self.bn(x))
        
        # Process with each dilation rate in parallel
        scale_features = []
        for conv in self.dilated_convs:
            scale_features.append(conv(x_activated))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=1)
        
        # Fuse scales
        fused = F.relu(self.fusion_bn(self.fusion_conv(multi_scale)))
        
        return fused
    
    def get_total_receptive_field(self) -> int:
        """Calculate the maximum receptive field across all scales."""
        max_rf = 0
        for conv in self.dilated_convs:
            rf = conv.get_receptive_field_size()
            max_rf = max(max_rf, rf)
        return max_rf


class AtrousSpatialPyramidPooling(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) from DeepLab.
    
    Captures multi-scale context by applying multiple dilated convolutions
    in parallel, along with global average pooling for scene-level features.
    
    Key innovation: Parallel processing of multiple scales instead of
    sequential, enabling better gradient flow and feature integration.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 256,
                 dilations: List[int] = [6, 12, 18]):
        super().__init__()
        
        self.dilations = dilations
        
        # 1x1 convolution for channel reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Parallel dilated convolutions
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            conv_block = nn.Sequential(
                DilatedConv2d(in_channels, out_channels, 3, dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.dilated_convs.append(conv_block)
        
        # Global average pooling branch for image-level features
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion convolution
        total_channels = out_channels * (len(dilations) + 2)  # +2 for 1x1 and global
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        h, w = x.size()[2:]
        
        # 1x1 convolution path
        conv_1x1_out = self.conv_1x1(x)
        
        # Dilated convolution paths
        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))
        
        # Global pooling path
        global_out = self.global_pool(x)
        global_out = F.interpolate(global_out, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate all paths
        all_features = [conv_1x1_out] + dilated_outputs + [global_out]
        concatenated = torch.cat(all_features, dim=1)
        
        # Final fusion
        return self.fusion(concatenated)


class DilatedResNet(nn.Module):
    """
    ResNet with dilated convolutions for maintaining spatial resolution.
    
    Replaces stride-2 convolutions and pooling with dilated convolutions
    in later layers, preserving spatial resolution while maintaining
    large receptive fields.
    
    Architecture principle: Early layers use standard convolutions for
    fine details, later layers use dilated convolutions for context.
    """
    
    def __init__(self, block_type, layers: List[int], num_classes: int = 1000,
                 output_stride: int = 8):
        super().__init__()
        
        self.output_stride = output_stride
        self.current_stride = 1
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.current_stride = 4
        
        # Residual layers with adaptive stride/dilation
        self.in_channels = 64
        self.layer1 = self._make_layer(block_type, 64, layers[0])
        self.layer2 = self._make_layer(block_type, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)
    
    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1):
        """Create a layer with adaptive stride/dilation based on output_stride."""
        
        # Determine if we need to use dilation instead of stride
        use_dilation = False
        dilation = 1
        
        if stride != 1 and self.current_stride * stride > self.output_stride:
            # Use dilation instead of stride to maintain resolution
            use_dilation = True
            dilation = stride
            stride = 1
        else:
            self.current_stride *= stride
        
        # First block (may downsample)
        if use_dilation:
            first_block = DilatedResidualBlock(self.in_channels, out_channels, 
                                             stride=stride, dilation=dilation)
        else:
            first_block = DilatedResidualBlock(self.in_channels, out_channels, 
                                             stride=stride)
        
        layers = [first_block]
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            if use_dilation:
                layers.append(DilatedResidualBlock(self.in_channels, out_channels, 
                                                 dilation=dilation))
            else:
                layers.append(DilatedResidualBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class DilatedResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions.
    
    Maintains residual learning benefits while enabling multi-scale
    context capture through configurable dilation rates.
    """
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dilation: int = 1):
        super().__init__()
        
        # Main path with dilated convolutions
        self.conv1 = DilatedConv2d(in_channels, out_channels, 3, dilation, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DilatedConv2d(out_channels, out_channels, 3, dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class WaveNetBlock(nn.Module):
    """
    WaveNet-style dilated convolution block for sequential data.
    
    Uses causal dilated convolutions with gated activations for
    powerful sequential modeling with exponential receptive field growth.
    
    Innovation: Gated activation = tanh(conv) ⊙ sigmoid(gate_conv)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2,
                 dilation: int = 1):
        super().__init__()
        
        self.dilation = dilation
        
        # Causal padding for autoregressive generation
        self.padding = (kernel_size - 1) * dilation
        
        # Main dilated convolution
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
        
        # Gate convolution for gated activation
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  padding=self.padding, dilation=dilation)
        
        # 1x1 convolution for residual connection
        self.res_conv = nn.Conv1d(out_channels, in_channels, 1)
        
        # 1x1 convolution for skip connection
        self.skip_conv = nn.Conv1d(out_channels, in_channels, 1)
    
    def forward(self, x):
        # Remove future information (causal)
        conv_out = self.conv(x)[:, :, :-self.padding] if self.padding > 0 else self.conv(x)
        gate_out = self.gate_conv(x)[:, :, :-self.padding] if self.padding > 0 else self.gate_conv(x)
        
        # Gated activation: tanh(conv) ⊙ sigmoid(gate)
        activation = torch.tanh(conv_out) * torch.sigmoid(gate_out)
        
        # Residual connection
        residual = self.res_conv(activation)
        
        # Skip connection for later aggregation
        skip = self.skip_conv(activation)
        
        return x + residual, skip


class WaveNetModel(nn.Module):
    """
    Complete WaveNet model with stacked dilated convolution blocks.
    
    Demonstrates how dilated convolutions revolutionized sequential modeling
    by providing exponential receptive field growth with constant parameters.
    """
    
    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 residual_channels: int = 32, skip_channels: int = 32,
                 num_blocks: int = 10, num_layers: int = 4):
        super().__init__()
        
        # Input projection
        self.input_conv = nn.Conv1d(in_channels, residual_channels, 1)
        
        # Dilated convolution blocks
        self.blocks = nn.ModuleList()
        for layer in range(num_layers):
            for block in range(num_blocks):
                dilation = 2 ** block  # Exponential dilation: 1, 2, 4, 8, 16, ...
                self.blocks.append(
                    WaveNetBlock(residual_channels, skip_channels, 
                               kernel_size=2, dilation=dilation)
                )
        
        # Output layers
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.Conv1d(skip_channels, out_channels, 1)
        ])
    
    def forward(self, x):
        x = self.input_conv(x)
        
        skip_connections = []
        
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Aggregate skip connections
        skip_out = sum(skip_connections)
        
        # Final processing
        out = F.relu(self.skip_convs[0](skip_out))
        out = self.skip_convs[1](out)
        
        return out


class ReceptiveFieldAnalyzer:
    """
    Analyze and visualize receptive fields in dilated convolution networks.
    
    Provides tools to understand how dilated convolutions affect the
    effective receptive field and multi-scale context capture.
    """
    
    @staticmethod
    def calculate_receptive_field(layers: List[Dict], input_size: int = 1) -> List[int]:
        """
        Calculate receptive field progression through network layers.
        
        Args:
            layers: List of layer configs [{'kernel': k, 'dilation': d, 'stride': s}, ...]
            input_size: Initial receptive field size
            
        Returns:
            List of receptive field sizes after each layer
        """
        rf_sizes = [input_size]
        current_rf = input_size
        current_stride = 1
        
        for layer in layers:
            kernel = layer.get('kernel', 3)
            dilation = layer.get('dilation', 1)
            stride = layer.get('stride', 1)
            
            # Effective kernel size with dilation
            effective_kernel = kernel + (kernel - 1) * (dilation - 1)
            
            # Update receptive field
            current_rf = current_rf + (effective_kernel - 1) * current_stride
            current_stride *= stride
            
            rf_sizes.append(current_rf)
        
        return rf_sizes
    
    @staticmethod
    def compare_architectures(standard_layers: List[Dict], 
                            dilated_layers: List[Dict]) -> Dict:
        """
        Compare receptive fields between standard and dilated architectures.
        
        Returns:
            Dictionary with comparison metrics
        """
        standard_rf = ReceptiveFieldAnalyzer.calculate_receptive_field(standard_layers)
        dilated_rf = ReceptiveFieldAnalyzer.calculate_receptive_field(dilated_layers)
        
        # Calculate parameter counts (simplified)
        standard_params = sum(l.get('kernel', 3)**2 * l.get('channels', 64) for l in standard_layers)
        dilated_params = sum(l.get('kernel', 3)**2 * l.get('channels', 64) for l in dilated_layers)
        
        return {
            'standard_rf': standard_rf,
            'dilated_rf': dilated_rf,
            'rf_improvement': dilated_rf[-1] / standard_rf[-1] if standard_rf[-1] > 0 else float('inf'),
            'standard_params': standard_params,
            'dilated_params': dilated_params,
            'param_efficiency': dilated_rf[-1] / dilated_params if dilated_params > 0 else 0
        }


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor using parallel dilated convolutions.
    
    Demonstrates how dilated convolutions enable simultaneous processing
    of multiple spatial scales for rich feature representations.
    """
    
    def __init__(self, in_channels: int, feature_channels: int = 64,
                 scales: List[int] = [1, 2, 4, 8, 16]):
        super().__init__()
        
        self.scales = scales
        
        # Parallel feature extractors for each scale
        self.scale_extractors = nn.ModuleList()
        for scale in scales:
            extractor = nn.Sequential(
                DilatedConv2d(in_channels, feature_channels, 3, dilation=scale),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True),
                DilatedConv2d(feature_channels, feature_channels, 3, dilation=scale),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True)
            )
            self.scale_extractors.append(extractor)
        
        # Cross-scale fusion
        total_channels = feature_channels * len(scales)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract features at each scale
        scale_features = []
        for extractor in self.scale_extractors:
            features = extractor(x)
            scale_features.append(features)
        
        # Concatenate and fuse
        combined = torch.cat(scale_features, dim=1)
        fused = self.fusion(combined)
        
        return fused, scale_features
    
    def get_scale_statistics(self, scale_features: List[torch.Tensor]) -> Dict:
        """
        Analyze statistics of features at different scales.
        
        Returns:
            Dictionary with per-scale feature statistics
        """
        stats = {}
        for i, (scale, features) in enumerate(zip(self.scales, scale_features)):
            stats[f'scale_{scale}'] = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'max': features.max().item(),
                'min': features.min().item(),
                'sparsity': (features == 0).float().mean().item()
            }
        
        return stats


class DilatedSegmentationHead(nn.Module):
    """
    Segmentation head using dilated convolutions for dense prediction.
    
    Demonstrates how dilated convolutions revolutionized semantic segmentation
    by maintaining spatial resolution while capturing global context.
    """
    
    def __init__(self, in_channels: int, num_classes: int, 
                 use_aspp: bool = True):
        super().__init__()
        
        self.use_aspp = use_aspp
        
        if use_aspp:
            # Use ASPP for multi-scale context
            self.context_module = AtrousSpatialPyramidPooling(in_channels)
            decoder_in_channels = 256
        else:
            # Simple dilated convolution sequence
            self.context_module = nn.Sequential(
                DilatedConv2d(in_channels, 256, 3, dilation=6),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                DilatedConv2d(256, 256, 3, dilation=12),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                DilatedConv2d(256, 256, 3, dilation=18),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            decoder_in_channels = 256
        
        # Final classification
        self.classifier = nn.Conv2d(decoder_in_channels, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        # Multi-scale context extraction
        context_features = self.context_module(x)
        
        # Apply dropout and classify
        context_features = self.dropout(context_features)
        output = self.classifier(context_features)
        
        return output


# Factory functions for common architectures
def create_deeplab_v3(num_classes: int = 21, backbone: str = 'resnet50') -> nn.Module:
    """Create a DeepLab v3 model with ASPP."""
    
    class DeepLabV3(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            
            # Simplified backbone (normally would use pre-trained ResNet)
            self.backbone = DilatedResNet(DilatedResidualBlock, [3, 4, 6, 3], 
                                        output_stride=16)
            
            # Remove classification head
            self.backbone.avgpool = nn.Identity()
            self.backbone.fc = nn.Identity()
            
            # ASPP head
            self.aspp = AtrousSpatialPyramidPooling(512, 256)
            
            # Final classifier
            self.classifier = nn.Conv2d(256, num_classes, 1)
        
        def forward(self, x):
            h, w = x.size()[2:]
            
            # Backbone feature extraction
            features = self.backbone(x)
            
            # Multi-scale context with ASPP
            aspp_features = self.aspp(features)
            
            # Classification
            output = self.classifier(aspp_features)
            
            # Upsample to original resolution
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            
            return output
    
    return DeepLabV3(num_classes)


def create_wavenet_model(input_dim: int = 256, output_dim: int = 256) -> WaveNetModel:
    """Create a WaveNet model for audio generation."""
    return WaveNetModel(input_dim, output_dim)


# Example usage and demonstrations
if __name__ == "__main__":
    print("🌊 Dilated Convolutions Implementation")
    print("=" * 50)
    
    # 1. Basic dilated convolution
    print("\n1. Basic Dilated Convolution:")
    dilated_conv = DilatedConv2d(64, 128, kernel_size=3, dilation=4)
    print(f"   Effective kernel size: {dilated_conv.effective_kernel_size}")
    print(f"   Parameters: {dilated_conv.get_parameter_count():,}")
    
    # 2. Multi-scale dilated block
    print("\n2. Multi-Scale Dilated Block:")
    multi_scale = MultiScaleDilatedBlock(64, 256, dilations=[1, 2, 4, 8])
    print(f"   Total receptive field: {multi_scale.get_total_receptive_field()}")
    
    # 3. ASPP demonstration
    print("\n3. Atrous Spatial Pyramid Pooling:")
    aspp = AtrousSpatialPyramidPooling(512, 256)
    
    # Test with sample input
    x = torch.randn(1, 512, 32, 32)
    aspp_out = aspp(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {aspp_out.shape}")
    print(f"   ✅ Resolution preserved!")
    
    # 4. Receptive field analysis
    print("\n4. Receptive Field Analysis:")
    analyzer = ReceptiveFieldAnalyzer()
    
    # Compare standard vs dilated architectures
    standard_layers = [
        {'kernel': 3, 'stride': 2, 'channels': 64},
        {'kernel': 3, 'stride': 2, 'channels': 128},
        {'kernel': 3, 'stride': 2, 'channels': 256}
    ]
    
    dilated_layers = [
        {'kernel': 3, 'dilation': 1, 'channels': 64},
        {'kernel': 3, 'dilation': 2, 'channels': 128},
        {'kernel': 3, 'dilation': 4, 'channels': 256}
    ]
    
    comparison = analyzer.compare_architectures(standard_layers, dilated_layers)
    print(f"   Standard RF: {comparison['standard_rf'][-1]}")
    print(f"   Dilated RF: {comparison['dilated_rf'][-1]}")
    print(f"   RF improvement: {comparison['rf_improvement']:.2f}x")
    
    # 5. WaveNet demonstration
    print("\n5. WaveNet Sequential Modeling:")
    wavenet = create_wavenet_model()
    
    # Test with sequential data
    seq_data = torch.randn(1, 256, 1000)  # Batch, channels, sequence length
    wavenet_out = wavenet(seq_data)
    print(f"   Input sequence: {seq_data.shape}")
    print(f"   Output sequence: {wavenet_out.shape}")
    
    # 6. DeepLab V3 segmentation
    print("\n6. DeepLab V3 Segmentation:")
    deeplab = create_deeplab_v3(num_classes=21)
    
    # Test with image
    image = torch.randn(1, 3, 256, 256)
    seg_output = deeplab(image)
    print(f"   Input image: {image.shape}")
    print(f"   Segmentation output: {seg_output.shape}")
    print(f"   ✅ Full resolution output!")
    
    # 7. Multi-scale feature extraction
    print("\n7. Multi-Scale Feature Extraction:")
    extractor = MultiScaleFeatureExtractor(3, 64, scales=[1, 2, 4, 8, 16])
    
    test_image = torch.randn(1, 3, 128, 128)
    fused_features, scale_features = extractor(test_image)
    
    print(f"   Input: {test_image.shape}")
    print(f"   Fused features: {fused_features.shape}")
    print(f"   Number of scales: {len(scale_features)}")
    
    # Analyze scale statistics
    stats = extractor.get_scale_statistics(scale_features)
    for scale_name, scale_stats in stats.items():
        print(f"   {scale_name}: mean={scale_stats['mean']:.3f}, std={scale_stats['std']:.3f}")
    
    print("\n🎯 Dilated Convolutions Summary:")
    print("   ✅ Multi-scale context without resolution loss")
    print("   ✅ Parameter efficiency with exponential RF growth")
    print("   ✅ Parallel processing of multiple scales")
    print("   ✅ Foundation for modern dense prediction tasks")
    print("   ✅ Sequential modeling breakthrough (WaveNet)")
    
    print("\n🚀 Ready for multi-scale context aggregation!")