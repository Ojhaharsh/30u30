# Day 10: ResNet V2 - Exercises

Explore improved ResNet architecture with pre-activation and identity mappings!

---

## Exercise 1: Pre-Activation vs Post-Activation ⭐⭐⭐

Compare the architectural differences between ResNet v1 and v2.

**Problem**: Implement both architectures and measure impact on training dynamics.

**Starting Code**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class PostActivationBlock(nn.Module):
    """ResNet v1: Conv -> BN -> ReLU (post-activation)"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.skip = nn.Identity() if stride == 1 and in_ch == out_ch else \
                   nn.Sequential(
                       nn.Conv2d(in_ch, out_ch, 1, stride),
                       nn.BatchNorm2d(out_ch)
                   )
    
    def forward(self, x):
        # TODO: Implement post-activation
        # conv -> bn -> relu -> conv -> bn -> add skip -> relu
        pass

class PreActivationBlock(nn.Module):
    """ResNet v2: BN -> ReLU -> Conv (pre-activation)"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        
        self.skip = nn.Identity() if stride == 1 and in_ch == out_ch else \
                   nn.Conv2d(in_ch, out_ch, 1, stride)
    
    def forward(self, x):
        # TODO: Implement pre-activation
        # bn -> relu -> conv -> bn -> relu -> conv -> add skip
        pass

def compare_activation_order():
    """Compare post-activation (v1) vs pre-activation (v2)"""
    
    # TODO: Build models with both block types
    # Train on CIFAR-10 for 30 epochs
    # Measure: convergence speed, final accuracy, training stability
    
    results = {}
    
    return results

results = compare_activation_order()
# Plot comparison
```

**Expected Results**:
- Post-activation (v1): Good but slightly slower convergence
- Pre-activation (v2): Faster convergence, better final accuracy
- Pre-activation enables gradient flow before adding skip connection
- Improvement: ~2-3% better accuracy with same depth

**Hints**:
1. Post-activation: activation applied after skip connection addition
2. Pre-activation: activation applied before convolution (on input)
3. Pre-activation makes network easier to optimize
4. v2 converges faster with same learning rate

**Grading Rubric**:
- ⭐⭐⭐ (7/10): Correctly implements both versions
- ⭐⭐⭐⭐ (9/10): Shows pre-activation trains faster
- ⭐⭐⭐⭐⭐ (10/10): Explains why pre-activation improves optimization

---

## Exercise 2: Identity Mapping Analysis ⭐⭐⭐⭐

Understand how pre-activation preserves identity information.

**Problem**: Analyze how well identity information flows through the network with different architectures.

**Starting Code**:
```python
def analyze_identity_preservation():
    """Measure how much identity information is preserved"""
    
    # For post-activation and pre-activation blocks:
    # 1. Forward pass through multiple blocks
    # 2. Measure: residual magnitude, identity preservation
    # 3. Visualize: how input affects each layer
    
    # Key metric: ||F(x)|| / ||x||
    # - If small: identity dominates
    # - If large: transformation dominates
    # - Pre-activation should have better balance
    
    # TODO: Forward hooks to capture:
    # - Input magnitude ||x||
    # - Residual magnitude ||F(x)||
    # - Output magnitude ||F(x) + x||
    
    # Plot histograms for analysis
    pass

analyze_identity_preservation()
```

**Expected Findings**:
- Pre-activation: Better identity preservation
- Residuals are sparser in v2
- Cleaner feature hierarchies with pre-activation
- Identity acts as better skip connection

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Measures identity vs residual magnitudes
- ⭐⭐⭐⭐⭐ (10/10): Explains architectural benefits mathematically

---

## Exercise 3: Batch Normalization Position ⭐⭐⭐⭐

Explore different BN placements in residual blocks.

**Problem**: Compare BN in different positions: pre-activation, post-activation, both, neither.

**Starting Code**:
```python
class NoBN(nn.Module):
    """No batch normalization"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Just convolutions
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride) if stride != 1 or in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        # TODO: forward pass without BN
        pass

class BNAfterConv(nn.Module):
    """BN after each convolution"""
    # TODO: Similar structure but add BN after conv

class BNBeforeConv(nn.Module):
    """BN before each convolution"""
    # TODO: Pre-activation style

def compare_bn_positions():
    """Compare different BN placements"""
    
    variants = {
        'No BN': NoBN,
        'BN after Conv': BNAfterConv,
        'BN before Conv': BNBeforeConv
    }
    
    # TODO: Train each variant
    # Measure: convergence, final accuracy, training stability
    
    return results

results = compare_bn_positions()
```

**Expected Results**:
- No BN: Unstable, poor convergence
- BN after Conv: Good, stable training
- BN before Conv (v2): Best convergence speed
- Pre-activation BN is optimal

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Tests all 3 BN placements
- ⭐⭐⭐⭐⭐ (10/10): Shows pre-activation BN is best

---

## Exercise 4: Gradient Flow Through Depth ⭐⭐⭐⭐⭐

Analyze how gradients flow with pre-activation architecture.

**Problem**: Measure gradient flow across network depth and show pre-activation preserves it better.

**Starting Code**:
```python
def analyze_gradient_flow():
    """Measure gradients at different depths"""
    
    models = {
        'PostActivation': build_model('post'),
        'PreActivation': build_model('pre')
    }
    
    # TODO: Forward and backward pass
    # Hook into each layer and measure:
    # - Input gradient magnitude
    # - Output gradient magnitude
    # - Gradient attenuation factor
    
    # Key insight: Pre-activation should have more uniform gradient norms
    
    # Plot: gradient norm vs layer depth
    # Pre-activation line should be flatter (better preservation)
    
    pass

analyze_gradient_flow()
```

**Expected Findings**:
- Post-activation: Gradient decay with depth
- Pre-activation: More uniform gradient norms
- Pre-activation: Gradients don't vanish
- Better optimization with pre-activation

**Hints**:
1. Register backward hooks to capture gradients
2. Calculate gradient attenuation: grad_layer_i / grad_layer_0
3. Expect roughly constant ratio for pre-activation
4. Post-activation should show exponential decay

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Measures gradient flow at multiple depths
- ⭐⭐⭐⭐⭐ (10/10): Shows pre-activation preserves gradients better

---

## Exercise 5: ResNet v2 vs Modern Architectures ⭐⭐⭐⭐⭐

Compare ResNet v2 with other modern architectures.

**Problem**: Train ResNet v2, EfficientNet concepts, Vision Transformer concepts on same task.

**Starting Code**:
```python
def compare_modern_architectures():
    """Compare ResNet-v2 with other approaches"""
    
    architectures = {
        'ResNet-v2': ResNetV2(depth=50),
        'Wide-ResNet': WideResNet(depth=50, width=2.0),
        'Dense-ResNet': DenseResNet(depth=50),  # ResNet + dense connections
    }
    
    # TODO: Train each for fair comparison
    # Same training time/epochs
    # Measure: accuracy, training speed, memory usage
    
    # Modern finding: Architecture matters less than:
    # 1. Proper training (learning rate schedule)
    # 2. Data augmentation
    # 3. Regularization
    # 4. Batch size
    
    pass

compare_modern_architectures()
```

**Expected Results**:
- ResNet v2: Solid baseline
- Wide-ResNet: Better with more parameters
- Dense connections: More efficient
- Training details matter more than architecture

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Compares v2 with at least 2 other architectures
- ⭐⭐⭐⭐⭐ (10/10): Shows training procedures matter more than architecture

---

## Bonus Challenge: Hardware-Efficient Design 🚀

Design ResNet v2 variants optimized for different devices.

```python
# TODO: Optimize for:
# 1. Mobile (MobileNet principles)
# 2. Edge devices (pruning, quantization)
# 3. Cloud (maximize accuracy at scale)
# 4. Specialized hardware (TPU, NPU)

# This connects to practical deployment!
```

---

## Summary

**Concepts Mastered**:
- ✅ Pre-activation vs post-activation design
- ✅ Batch normalization placement
- ✅ Identity mapping through deep networks
- ✅ Gradient flow analysis
- ✅ Architecture design trade-offs

**Aha! Moments**:
1. Activation order matters for optimization
2. Pre-activation enables better gradient flow
3. Identity mappings are key to scaling
4. Architecture details affect convergence speed
5. ResNet v2 refined v1's successful formula

**Evolution of ResNets**:
- ResNet (2015): Skip connections revolutionized deep learning
- ResNet v2 (2016): Pre-activation improved on original
- Wide-ResNet (2016): Width beats depth
- Modern (2020+): Hybrid architectures combining best ideas

You've now mastered 2.5 generations of ResNet evolution! 🚀
