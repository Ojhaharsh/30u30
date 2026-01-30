# Day 11: Dilated Convolutions - Exercises

Explore receptive field expansion and multi-scale context aggregation!

---

## Exercise 1: Receptive Field Comparison ⭐⭐⭐

Understand how dilation affects receptive field size.

**Problem**: Compare standard conv, dilated conv, and large kernels. Measure effective receptive field.

**Starting Code**:
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def calculate_receptive_field(kernel_size, stride, dilation, num_layers):
    """Calculate effective receptive field (theory)"""
    rf = 1
    for _ in range(num_layers):
        rf = rf + (kernel_size - 1) * dilation * stride
    return rf

class StandardConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        return self.blocks(x)

class DilatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=3):
        super().__init__()
        # TODO: Create blocks with increasing dilation
        # Block 1: dilation=1
        # Block 2: dilation=2
        # Block 3: dilation=4
        pass
    
    def forward(self, x):
        pass

class LargeKernelConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=3):
        super().__init__()
        # TODO: Use larger kernels (7x7) instead of dilation
        pass

def visualize_receptive_field():
    """Visualize which input pixels affect output"""
    
    # Create test input
    x = torch.zeros(1, 1, 32, 32)
    x[0, 0, 15, 15] = 1.0  # Mark center pixel
    
    models = {
        'Standard Conv': StandardConv(1, 1),
        'Dilated Conv': DilatedConv(1, 1),
        'Large Kernel': LargeKernelConv(1, 1)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            # Forward pass with gradient to see input influence
            x_test = x.clone().detach().requires_grad_(True)
            out = model(x_test)
            out.sum().backward()
        
        # Visualize gradient magnitude (receptive field)
        receptive_field = x_test.grad.abs()
        
        ax = axes[idx]
        im = ax.imshow(receptive_field[0, 0].cpu().numpy(), cmap='hot')
        ax.set_title(f'{name}\nReceptive Field Size')
        plt.colorbar(im, ax=ax, label='Gradient Magnitude')
    
    plt.tight_layout()
    plt.show()

def compare_receptive_fields():
    """Compare receptive field sizes"""
    
    # TODO: Calculate theoretical receptive field sizes
    # Standard conv: 3x3 kernel, 3 layers → RF = 1 + 2*1 + 2*1 + 2*1 = 7
    # Dilated conv: 3x3 kernel, dilations [1,2,4] → RF = much larger
    # Large kernel: 7x7 kernel → RF = 7 directly
    
    # Plot: receptive field vs parameter count
    # Show trade-offs
    
    pass

visualize_receptive_field()
```

**Expected Results**:
- Standard conv (3x3, 3 layers): RF = 7×7
- Dilated conv (3x3, dilations [1,2,4]): RF = 15×15
- Large kernel (7x7, 3 layers): RF = ~15×15
- Dilated conv achieves larger RF with fewer parameters!

**Hints**:
1. Use gradient backprop to visualize receptive field
2. Calculate receptive field formula: RF = 1 + sum((K-1)*D*S)
3. Dilated conv is more parameter efficient
4. Both achieve large receptive fields with less computation

**Grading Rubric**:
- ⭐⭐⭐ (7/10): Compares receptive field sizes correctly
- ⭐⭐⭐⭐ (9/10): Visualizes receptive field with gradients
- ⭐⭐⭐⭐⭐ (10/10): Shows parameter efficiency of dilated conv

---

## Exercise 2: Semantic Segmentation with Dilated Convolutions ⭐⭐⭐⭐

Build a simple segmentation network using dilated convolutions.

**Problem**: Implement a U-Net-style architecture with dilated convolutions. Compare with standard convolutions.

**Starting Code**:
```python
class DilatedUNet(nn.Module):
    """U-Net with dilated convolutions"""
    def __init__(self):
        super().__init__()
        
        # Encoder: standard convolutions
        self.enc1 = self.conv_block(3, 64, 1)
        self.enc2 = self.conv_block(64, 128, 2)
        
        # Bottleneck: dilated convolutions
        # TODO: Use dilated conv to capture context
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: standard convolutions with skip connections
        self.dec2 = self.conv_block(256 + 128, 128, 1)
        self.dec1 = self.conv_block(128 + 64, 64, 1)
        
        # Output
        self.out = nn.Conv2d(64, 1, 1)  # Single channel for segmentation
    
    def conv_block(self, in_ch, out_ch, stride):
        # TODO: Standard conv block with BN and ReLU
        pass
    
    def forward(self, x):
        # TODO: Implement encoder-decoder with skip connections
        pass

def compare_segmentation_architectures():
    """Compare dilated vs standard convolution for segmentation"""
    
    # TODO: Train both architectures on simple segmentation task
    # Measure: convergence, final dice score, memory usage
    
    results = {}
    
    return results

results = compare_segmentation_architectures()
```

**Expected Results**:
- Standard U-Net: Good accuracy but limited context
- Dilated U-Net: Better accuracy with larger context window
- Dilated conv improves segmentation by 3-5%
- Both use similar memory

**Hints**:
1. Dilated conv maintains spatial resolution (important for segmentation)
2. Skip connections preserve fine details
3. Larger receptive field captures global context
4. Combine local (conv) and global (dilated conv) information

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Implements dilated U-Net correctly
- ⭐⭐⭐⭐ (9/10): Shows dilated conv improves segmentation
- ⭐⭐⭐⭐⭐ (10/10): Analyzes memory/speed/accuracy trade-offs

---

## Exercise 3: Atrous Spatial Pyramid Pooling ⭐⭐⭐⭐⭐

Implement ASPP module that uses multiple dilation rates.

**Problem**: Combine dilated convolutions with different dilation rates to capture multi-scale context.

**Starting Code**:
```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # 1x1 conv
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, 1)
        
        # Dilated convolutions with different rates
        self.dilated_3x3_r6 = nn.Conv2d(in_ch, out_ch, 3, dilation=6, padding=6)
        self.dilated_3x3_r12 = nn.Conv2d(in_ch, out_ch, 3, dilation=12, padding=12)
        self.dilated_3x3_r18 = nn.Conv2d(in_ch, out_ch, 3, dilation=18, padding=18)
        
        # Image-level features (global pooling)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
            # TODO: Upsample back to input spatial size
        )
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # TODO: Apply all branches and concatenate
        # Branches: 1x1, dilated 3x3 (r=6,12,18), image pool
        # Concatenate and project
        pass

def build_deeplab_lite():
    """Simplified DeepLabv3 with ASPP"""
    
    class DeepLabLite(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder (ResNet backbone)
            self.backbone = nn.Sequential(
                # Standard convolutions with stride=2 to downsample
            )
            
            # ASPP module
            self.aspp = ASPP(256, 256)
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1)  # Segmentation output
            )
        
        def forward(self, x):
            # TODO: Backbone -> ASPP -> Decoder
            pass
    
    return DeepLabLite()

def test_aspp_module():
    """Test ASPP on segmentation task"""
    
    # TODO: Train model with ASPP
    # Measure: accuracy, receptive field, parameter count
    
    pass

test_aspp_module()
```

**Expected Results**:
- ASPP captures multi-scale features effectively
- Combines local (1x1) and global (large dilation) context
- Improves segmentation by 5-10%
- More parameters but better use of receptive field

**Hints**:
1. ASPP = "Atrous Spatial Pyramid Pooling"
2. Different dilation rates capture different scales
3. Image pool branch captures global context
4. Concatenate all branches for multi-scale features
5. This is the key innovation in DeepLabv3!

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Correctly implements ASPP module
- ⭐⭐⭐⭐⭐ (10/10): Shows ASPP significantly improves segmentation

---

## Exercise 4: Multi-Scale Feature Fusion ⭐⭐⭐⭐⭐

Combine dilated convolutions with multi-resolution features.

**Problem**: Implement feature pyramid with dilated convolutions at each level.

**Starting Code**:
```python
class DilatedFeaturePyramid(nn.Module):
    """Feature pyramid with dilated convolutions"""
    def __init__(self, in_ch):
        super().__init__()
        
        # Multiple scales with different dilations
        self.scale_1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, dilation=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.scale_2 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True)
        )
        
        self.scale_3 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True)
        )
        
        self.scale_4 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, dilation=8, padding=8),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Conv2d(256, 64, 1)
    
    def forward(self, x):
        # TODO: Apply all scales and fuse
        pass

def test_multi_scale_fusion():
    """Test multi-scale feature fusion"""
    
    # TODO: Train segmentation model with multi-scale fusion
    # Compare with: single scale, concatenation, attention fusion
    
    pass

test_multi_scale_fusion()
```

**Expected Results**:
- Multi-scale helps capture fine and coarse details
- Dilation rates 1, 2, 4, 8 give good coverage
- Learned fusion better than simple concatenation
- Improvement: 3-5% accuracy

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Implements and tests multi-scale feature fusion
- ⭐⭐⭐⭐⭐ (10/10): Shows learned fusion outperforms concatenation

---

## Exercise 5: Receptive Field vs Accuracy Trade-off ⭐⭐⭐⭐⭐

Analyze the relationship between receptive field and task performance.

**Problem**: Train networks with different dilation strategies. Plot accuracy vs effective receptive field.

**Starting Code**:
```python
def measure_receptive_field_impact():
    """Measure how receptive field affects segmentation accuracy"""
    
    # Define networks with different receptive fields
    configurations = [
        {'name': 'RF=7', 'dilations': [1, 1, 1]},
        {'name': 'RF=15', 'dilations': [1, 2, 4]},
        {'name': 'RF=31', 'dilations': [1, 2, 4, 8]},
        {'name': 'RF=63', 'dilations': [1, 2, 4, 8, 16]},
    ]
    
    results = []
    
    for config in configurations:
        # Build model with specified dilations
        # Train on segmentation task
        # Measure: accuracy, convergence speed, memory usage
        
        # Calculate theoretical receptive field
        # Plot results
        
        pass
    
    # Visualization:
    # X-axis: Receptive Field Size
    # Y-axis: Segmentation Accuracy
    # Expected: Accuracy increases with RF up to image size
    # Then diminishing returns
    
    pass

measure_receptive_field_impact()
```

**Expected Results**:
- Small RF (7): Underfits, low accuracy
- Medium RF (15-31): Good balance, peak accuracy
- Large RF (63+): Slight improvement or plateau
- Optimal ≈ RF = image size (for full context)

**Hints**:
1. Receptive field must exceed object size for good segmentation
2. Larger receptive field = better global context
3. But too large = unnecessary computation
4. Task-dependent: small objects need smaller RF, large objects need larger RF

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Tests multiple receptive field sizes
- ⭐⭐⭐⭐⭐ (10/10): Discovers optimal RF for task

---

## Bonus Challenge: Mobile-Friendly Dilated Convolutions 🚀

Design efficient dilated conv for mobile devices!

```python
# TODO: Implement:
# 1. Grouped dilated convolutions (reduce parameters)
# 2. Depthwise separable dilated convolutions
# 3. Quantization-aware dilated conv
# 4. Benchmark on mobile hardware

# Goal: Achieve 90% of accuracy with 10% of parameters!
```

---

## Summary

**Concepts Mastered**:
- ✅ Receptive field expansion with dilation
- ✅ Dilated convolutions in practice
- ✅ Semantic segmentation architectures
- ✅ Multi-scale feature fusion with ASPP
- ✅ Trade-offs: receptive field vs computation

**Aha! Moments**:
1. Dilation expands receptive field without losing resolution
2. Multi-scale features capture both local and global context
3. ASPP enables state-of-the-art segmentation
4. Context matters more than high resolution pixels
5. Receptive field is a key design parameter

**Real-World Applications**:
- Semantic segmentation (road, buildings, etc.)
- Instance segmentation (individual objects)
- Medical image segmentation
- Video understanding with temporal context
- Even used in modern language models (Transformers have different context!)

**Historical Evolution**:
- Standard Conv (ImageNet): Local context only
- ResNet (2015): Better optimization with skip connections
- Dilated Conv (2016): Global context without resolution loss
- ASPP/DeepLabv3 (2017): Multi-scale context fusion
- Modern (2020+): Hybrid approaches (dilated conv + attention)

You've now mastered spatial context and receptive field design! 🚀
