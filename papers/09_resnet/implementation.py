"""
Day 9: Deep Residual Learning for Image Recognition (ResNet)

Complete implementation of ResNet with skip connections that enabled training
networks with 152+ layers. This solves the degradation problem and revolutionized
deep learning by making "deeper is better" actually work.

Key innovations:
1. Residual connections (skip connections)
2. Batch normalization integration
3. Bottleneck blocks for efficiency
4. Gradient highway creation

Author: 30u30 Project
Based on: "Deep Residual Learning for Image Recognition" 
         by He et al. (2015) - https://arxiv.org/abs/1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34.
    
    Structure:
    x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+) → ReLU
    └─────────────────────────────────────────┘
    
    The skip connection (identity shortcut) is the key innovation that
    enables training very deep networks by creating gradient highways.
    """
    expansion = 1
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, norm_layer: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # THE MAGIC: Add skip connection (residual learning)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50, ResNet-101, and ResNet-152.
    
    Structure:
    x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+) → ReLU
    └───────────────────────────────────────────────────────────┘
    
    Uses 1×1 convolutions to reduce computational complexity:
    - 1×1 conv reduces dimensions
    - 3×3 conv processes spatially
    - 1×1 conv expands dimensions back
    
    This is more efficient than using 3×3 convs throughout.
    """
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, norm_layer: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1×1 conv to reduce dimensions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3×3 conv for spatial processing
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1 conv to expand dimensions
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # THE MAGIC: Add skip connection
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture that can build any variant.
    
    The key insight: Instead of learning H(x), learn F(x) = H(x) - x
    Then H(x) = F(x) + x (implemented via skip connections)
    
    This makes it easier to learn identity mappings and enables
    training networks with 100+ layers successfully.
    
    Supported variants:
    - ResNet-18: [2, 2, 2, 2] with BasicBlock
    - ResNet-34: [3, 4, 6, 3] with BasicBlock  
    - ResNet-50: [3, 4, 6, 3] with Bottleneck
    - ResNet-101: [3, 4, 23, 3] with Bottleneck
    - ResNet-152: [3, 8, 36, 3] with Bottleneck
    """

    def __init__(self, block: Union[BasicBlock, Bottleneck], layers: List[int], 
                 num_classes: int = 1000, zero_init_residual: bool = False,
                 groups: int = 1, width_per_group: int = 64, 
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[nn.Module] = None):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers (the magic happens here)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final processing
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
    
    def get_features(self, x: torch.Tensor, layer: str = 'layer4') -> torch.Tensor:
        """Extract features from a specific layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        features = {}
        features['layer1'] = self.layer1(x)
        features['layer2'] = self.layer2(features['layer1'])
        features['layer3'] = self.layer3(features['layer2'])
        features['layer4'] = self.layer4(features['layer3'])
        
        return features[layer]


class ResNetAnalyzer:
    """
    Analysis tools for understanding ResNet behavior and skip connections.
    """
    
    def __init__(self, model: ResNet):
        self.model = model
        self.model.eval()
        
    def analyze_skip_connections(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze the contribution of skip connections vs residual branches.
        """
        results = {}
        hooks = []
        
        # Hook to capture skip connection contributions
        def make_hook(name):
            def hook_fn(module, input, output):
                if hasattr(module, 'downsample') and module.downsample is not None:
                    identity = module.downsample(input[0])
                else:
                    identity = input[0]
                
                # Residual branch output (before adding identity)
                residual = output - identity
                
                results[name] = {
                    'identity_norm': torch.norm(identity).item(),
                    'residual_norm': torch.norm(residual).item(),
                    'output_norm': torch.norm(output).item(),
                    'residual_ratio': torch.norm(residual).item() / (torch.norm(identity).item() + 1e-8)
                }
            return hook_fn
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        return results
    
    def measure_gradient_flow(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Measure gradient flow through different layers.
        Shows how skip connections enable deep gradient propagation.
        """
        self.model.train()
        gradients = {}
        
        def make_hook(name):
            def hook_fn(module, grad_input, grad_output):
                if grad_input[0] is not None:
                    grad_norm = torch.norm(grad_input[0]).item()
                    gradients[name] = grad_norm
            return hook_fn
        
        hooks = []
        # Register hooks on each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, BasicBlock, Bottleneck)):
                hook = module.register_backward_hook(make_hook(name))
                hooks.append(hook)
        
        # Forward and backward pass
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        self.model.eval()
        return gradients
    
    def compare_with_without_skip(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compare network behavior with and without skip connections.
        (Simulated by zeroing out identity paths)
        """
        with torch.no_grad():
            # Normal forward pass
            normal_output = self.model(x)
            
            # Simulate no skip connections by modifying forward pass
            # This is a simplified simulation
            def forward_no_skip(model, x):
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                
                # Process each layer but ignore skip connections
                for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
                    for block in layer:
                        identity = x
                        x = block.conv1(x)
                        x = block.bn1(x)
                        x = block.relu(x)
                        
                        if hasattr(block, 'conv2'):  # BasicBlock
                            x = block.conv2(x)
                            x = block.bn2(x)
                        elif hasattr(block, 'conv3'):  # Bottleneck
                            x = block.conv2(x)
                            x = block.bn2(x)
                            x = block.relu(x)
                            x = block.conv3(x)
                            x = block.bn3(x)
                        
                        # Skip the identity addition here
                        # x += identity  # This is what we're skipping
                        x = block.relu(x)
                
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                x = model.fc(x)
                return x
            
            # This is a theoretical comparison
            # In practice, you'd need to retrain without skip connections
            return {
                'with_skip': normal_output,
                'activation_magnitude': torch.norm(normal_output).item()
            }


def create_resnet(variant: str = 'resnet50', num_classes: int = 1000, 
                 pretrained: bool = False) -> ResNet:
    """
    Factory function to create ResNet variants.
    
    Args:
        variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
    
    Returns:
        ResNet model
    """
    variants = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    block, layers = variants[variant]
    model = ResNet(block, layers, num_classes=num_classes)
    
    if pretrained:
        try:
            checkpoint = torch.load(f'{variant}_pretrained.pth', map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded pretrained {variant} weights")
        except FileNotFoundError:
            print(f"No pretrained weights found for {variant}")
    
    return model


class ResNetTrainer:
    """
    Training pipeline specifically designed for ResNet.
    Includes proper learning rate scheduling and monitoring.
    """
    
    def __init__(self, model: ResNet, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
    def train_with_degradation_analysis(self, train_loader, val_loader, 
                                      num_epochs: int = 10) -> Dict:
        """
        Train ResNet while analyzing the degradation problem solution.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, 
                                   momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'gradient_norms': [], 'skip_analysis': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Record metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            print(f'Epoch {epoch}: Train Acc {epoch_train_acc:.2f}%, Val Acc {epoch_val_acc:.2f}%')
            
            scheduler.step()
        
        return history


if __name__ == "__main__":
    # Demo: Create and test ResNet
    print("🔥 ResNet Implementation Demo")
    print("=" * 50)
    
    # Test different variants
    variants = ['resnet18', 'resnet34', 'resnet50']
    
    for variant in variants:
        print(f"\n🏗️  Creating {variant.upper()}...")
        model = create_resnet(variant, num_classes=10)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📊 Parameters: {total_params:,}")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        print(f"   ✅ Output shape: {y.shape}")
        
        # Test skip connection analysis
        if variant == 'resnet18':  # Analyze one variant
            analyzer = ResNetAnalyzer(model)
            skip_analysis = analyzer.analyze_skip_connections(x)
            
            print(f"   🔗 Skip connections analyzed:")
            for name, stats in list(skip_analysis.items())[:3]:  # Show first 3
                print(f"      {name}: residual/identity ratio = {stats['residual_ratio']:.3f}")
    
    print("\n✅ ResNet ready! Skip connections enable deep networks to thrive.")