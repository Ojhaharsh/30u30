# Day 9: ResNet - Exercises

Master skip connections and the architecture that enables arbitrarily deep networks!

---

## Exercise 1: The Vanishing Gradient Problem ⭐⭐⭐

Understand why deep networks fail without skip connections.

**Problem**: Compare training 20-layer vs 56-layer networks WITHOUT skip connections. Show that deeper isn't always better without ResNets.

**Starting Code**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class PlainConvNet(nn.Module):
    """Plain CNN without skip connections"""
    def __init__(self, depth=20):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create multiple residual blocks
        layers = []
        in_channels = 64
        for i in range(depth):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            ))
        
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        # TODO: Implement forward pass
        # - Apply conv1, bn1, relu, maxpool
        # - Pass through all layers
        # - Apply avgpool and fc
        pass

def compare_depth_impact():
    """Compare 20-layer vs 56-layer networks"""
    
    depths = [20, 56]
    results = {}
    
    # TODO: Train both models for 20 epochs
    # Track: training loss, validation accuracy, gradient norms
    # Measure: convergence speed and final accuracy
    
    return results

results = compare_depth_impact()
# Plot comparison
```

**Expected Results**:
- 20 layers: Trains well, converges to ~75% accuracy
- 56 layers (plain): Trains poorly, converges to ~65% accuracy (counterintuitive!)
- Gradient magnitude decreases exponentially with depth
- This is the "degradation problem" that ResNets solve

**Hints**:
1. Use `hook` to capture gradients at different layers
2. Plain networks suffer from vanishing gradients due to long backprop paths
3. Track gradient norm at first layer vs last layer
4. Notice deeper network actually performs worse!

**Grading Rubric**:
- ⭐⭐⭐ (7/10): Shows deeper network is worse without skip connections
- ⭐⭐⭐⭐ (9/10): Includes gradient analysis showing vanishing gradients
- ⭐⭐⭐⭐⭐ (10/10): Shows mathematical relationship: grad_norm ∝ (0.9)^depth

---

## Exercise 2: Skip Connections Implementation ⭐⭐⭐⭐

Implement ResNet blocks and see the magic.

**Problem**: Build a ResNet with skip connections and compare directly with plain CNN.

**Starting Code**:
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # TODO: Implement skip connection
        # Handle dimension mismatch with 1x1 convolution when needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # TODO: Implement forward pass
        # out = conv2(bn1(relu(conv1(x))))
        # Then add skip connection: out += shortcut(x)
        # Finally apply relu
        pass

class ResNet(nn.Module):
    def __init__(self, depth=56, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # TODO: Create residual blocks
        num_blocks = depth // 2
        self.layer1 = self._make_layer(ResidualBlock, 64, 64, num_blocks, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

def compare_plain_vs_resnet():
    # TODO: Train both Plain CNN (56 layers) and ResNet (56 layers)
    # Compare: convergence speed, final accuracy, gradient flow
    
    results = {}
    
    return results

results = compare_plain_vs_resnet()
```

**Expected Results**:
- Plain 56L: ~65% accuracy, training unstable
- ResNet 56L: ~85% accuracy, trains smoothly
- Gradient flow is preserved with skip connections
- Deeper networks now actually get better!

**Hints**:
1. Skip connections: `out = self.conv(x) + x`
2. When dimensions mismatch, use 1x1 conv in shortcut
3. Relu after adding skip connection
4. This is the core ResNet innovation!

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Correctly implements ResNet blocks and shows improvement
- ⭐⭐⭐⭐ (9/10): Compares gradient flow between plain and ResNet
- ⭐⭐⭐⭐⭐ (10/10): Shows that ResNet enables much deeper networks (100+) layers

---

## Exercise 3: Residual vs Identity Learning ⭐⭐⭐⭐

Understand the philosophy: learn residuals, not absolute values.

**Problem**: Analyze what features are learned in skip connections vs main path.

**Starting Code**:
```python
def analyze_residual_learning():
    """Understand what residuals vs identity capture"""
    
    # Train ResNet and track outputs
    model = ResNet(depth=56)
    
    # TODO: Forward pass through trained model
    # Capture: x, residual_path(x), output
    # Calculate: residual = output - x
    # Analyze: what is residual learning capturing?
    
    # Visualize residuals at different depths
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    # TODO: For several layers, show:
    # 1. Input activations distribution
    # 2. Residual (F(x)) distribution
    # 3. Output (F(x) + x) distribution
    # Compare: mean, std, range
    
    pass

analyze_residual_learning()
```

**Expected Findings**:
- Early layers: Small residuals (fine-tuning identity)
- Mid layers: Medium residuals (significant transformations)
- Late layers: Small residuals again (refinement)
- Residuals typically much smaller than inputs/outputs
- This explains why skip connections work: learning becomes easier!

**Hints**:
1. Hook into intermediate layers to capture activations
2. Calculate residual as `F(x) = output - x` before skip connection
3. Compare distributions using histograms
4. Residuals should be roughly Gaussian, zero-centered

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Visualizes and analyzes residual distributions
- ⭐⭐⭐⭐ (9/10): Explains why residuals are smaller than full outputs
- ⭐⭐⭐⭐⭐ (10/10): Discovers that residuals are sparse (many zeros)

---

## Exercise 4: Scaling to Very Deep Networks ⭐⭐⭐⭐

Build and train ResNet-50, ResNet-101, ResNet-152.

**Problem**: Show that skip connections enable arbitrarily deep networks.

**Starting Code**:
```python
class Bottleneck(nn.Module):
    """Bottleneck block for deeper ResNets"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1x1 reduce -> 3x3 -> 1x1 expand
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        # TODO: Implement forward pass with skip connection
        pass

class DeepResNet(nn.Module):
    def __init__(self, depths=[3, 4, 6, 3], num_classes=10):
        """depths specify # of bottleneck blocks in each stage"""
        super().__init__()
        # TODO: Build ResNet with specified depths
        # Example: ResNet-50 has depths=[3, 4, 6, 3]
        # Example: ResNet-101 has depths=[3, 4, 23, 3]
        # Example: ResNet-152 has depths=[3, 8, 36, 3]
        pass

def compare_depths():
    # TODO: Train ResNet-50, ResNet-101, ResNet-152
    # Measure: training time, final accuracy, convergence speed
    
    # Plot: accuracy vs number of layers
    # All should converge well due to skip connections!
    
    pass

compare_depths()
```

**Expected Results**:
- ResNet-50: ~80% accuracy, stable training
- ResNet-101: ~82% accuracy, slightly slower but still converges
- ResNet-152: ~83% accuracy, deepest still trains well
- All converge smoothly unlike plain networks

**Hints**:
1. Bottleneck blocks reduce memory and computation
2. Depths = [3, 4, 6, 3] means 4 stages with 3, 4, 6, 3 blocks each
3. Each stage reduces spatial dimension and increases channels
4. Still add batch norm after each conv

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Implements bottleneck blocks and builds deep ResNets
- ⭐⭐⭐⭐ (9/10): Trains multiple depths and shows all converge
- ⭐⭐⭐⭐⭐ (10/10): Shows accuracy improves with depth (up to ~152 layers)

---

## Exercise 5: Skip Connection Variations ⭐⭐⭐⭐⭐

Experiment with different skip connection designs.

**Problem**: Compare different ways of implementing skip connections.

**Starting Code**:
```python
class Identity Skip(nn.Module):
    """x + F(x) - direct skip"""
    pass

class ChannelWiseWeighting(nn.Module):
    """x + w * F(x) - learned weight per channel"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels) * 0.5)
    
    def forward(self, x, f_x):
        # TODO: Multiply F(x) by learned weights before adding
        pass

class AttentionSkip(nn.Module):
    """x + attention(x) * F(x) - spatial attention"""
    def __init__(self, channels):
        super().__init__()
        # TODO: Implement SE-Net style attention
        pass

class DenseSkip(nn.Module):
    """x + F(x) + G(x) - multiple paths"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Create multiple residual paths
        pass

def compare_skip_variants():
    # TODO: Train ResNet-50 with different skip variants
    # Measure: convergence speed, final accuracy, gradient flow
    
    variants = {
        'Identity Skip': Identity Skip(),
        'Channel Weighting': ChannelWiseWeighting(64),
        'Attention Skip': AttentionSkip(64),
        'Dense Skip': DenseSkip(64, 64)
    }
    
    results = {}
    
    return results

results = compare_skip_variants()
```

**Expected Results**:
- Identity skip: Baseline (already very good!)
- Channel weighting: Modest improvement
- Attention skip: Better feature selection
- Dense skip: Better feature mixing, slight improvement
- All variations preserve trainability

**Hints**:
1. Standard identity skip is already near-optimal
2. Learned weights can help prioritize features
3. Attention mechanisms can focus on important features
4. More connections = more gradient paths

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Implements multiple skip variants and compares
- ⭐⭐⭐⭐⭐ (10/10): Discovers that identity skip is surprisingly good baseline

---

## Bonus Challenge: Neural Architecture Search 🚀

Automatically discover optimal skip connection patterns!

```python
# TODO: Use evolutionary algorithm to optimize:
# - Which layers get skip connections?
# - Which layers to combine?
# - Skip connection strengths?

# This is the basis for NAS and AutoML!
```

---

## Summary

**Concepts Mastered**:
- ✅ Vanishing gradients in deep networks
- ✅ Skip connections solve gradient flow
- ✅ Residual learning philosophy
- ✅ Building arbitrarily deep networks (100+ layers!)
- ✅ Skip connection variants and improvements

**Aha! Moments**:
1. Deeper ≠ Better without skip connections
2. Skip connections make optimization easier
3. Residuals are typically much smaller than outputs
4. Modern networks use skip connections everywhere
5. This enables training 1000-layer networks!

**Practical Insight**: Skip connections are a fundamental tool in modern deep learning. They appear in ResNets, Transformers, U-Nets, and many others!
