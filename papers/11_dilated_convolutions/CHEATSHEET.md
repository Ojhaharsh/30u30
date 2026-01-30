# Dilated Convolutions Cheatsheet 📋

Quick reference for Multi-Scale Context without Losing Resolution

---

## The Big Idea (30 seconds)

Dilated convolutions expand the **receptive field** without losing spatial resolution by inserting "holes" (zeros) between filter weights. Instead of downsampling to see larger patterns, you can see them at full resolution!

- **Regular Conv**: Dense sampling, small receptive field
- **Dilated Conv**: Sparse sampling, large receptive field  
- **Result**: Multi-scale context aggregation without resolution loss

**Magic formula**: `receptive_field = (kernel_size - 1) × dilation + 1`

---

## Architecture: Multi-Scale Context Aggregation

### Dilation Pattern Examples
```
Dilation=1 (regular):    Dilation=2:        Dilation=4:
1 1 1                   1 0 1 0 1          1 0 0 0 1 0 0 0 1
1 1 1                   0 0 0 0 0          0 0 0 0 0 0 0 0 0
1 1 1                   1 0 1 0 1          0 0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0 0
Receptive field: 3x3    Receptive field: 5x5    Receptive field: 9x9
```

### ASPP (Atrous Spatial Pyramid Pooling)
```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Multiple dilation rates in parallel
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3x3_d6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3x3_d12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv3x3_d18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # Process at multiple scales simultaneously
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_d6(x)
        feat3 = self.conv3x3_d12(x)
        feat4 = self.conv3x3_d18(x)
        feat5 = F.interpolate(self.global_pool(x), size=x.shape[2:], mode='bilinear')
        
        # Concatenate multi-scale features
        return torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
```

---

## Quick Start

### Basic Dilated Convolution
```python
import torch.nn as nn

# Regular convolution
conv_regular = nn.Conv2d(64, 64, kernel_size=3, padding=1)

# Dilated convolution (2x larger receptive field)
conv_dilated = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

# Multi-scale parallel processing
conv_multi = nn.ModuleList([
    nn.Conv2d(64, 64, 3, padding=1, dilation=1),   # Fine details
    nn.Conv2d(64, 64, 3, padding=2, dilation=2),   # Medium context
    nn.Conv2d(64, 64, 3, padding=4, dilation=4),   # Large context
])
```

### Semantic Segmentation Application
```python
# DeepLab-style segmentation network
class DilatedSegNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.aspp = ASPP(2048, 256)
        self.classifier = nn.Conv2d(256 * 5, num_classes, 1)
        
    def forward(self, x):
        features = self.backbone.features(x)
        multi_scale = self.aspp(features)
        logits = self.classifier(multi_scale)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear')
```

---

## Key Applications

### 1. Semantic Segmentation
```python
# Dense prediction without losing resolution
class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace downsampling with dilation
        self.conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=2, dilation=2)  # Was stride=2
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=4, dilation=4) # Was stride=2
        self.conv3 = nn.Conv2d(256, 512, 3, stride=1, padding=8, dilation=8) # Was stride=2
```

### 2. WaveNet (Audio/Text Modeling)
```python
class WaveNetBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, 2*channels, 2, padding=dilation, dilation=dilation)
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        # Gated activation
        dilated = self.dilated_conv(x)
        filter_gate, gate = dilated.chunk(2, dim=1)
        gated = torch.tanh(filter_gate) * torch.sigmoid(gate)
        
        # Residual connection
        residual = self.residual_conv(gated)
        return x + residual, gated  # Skip connection + output
```

### 3. Multi-Scale Feature Extraction
```python
class MultiScaleExtractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Exponentially increasing dilation rates
        self.scales = nn.ModuleList([
            nn.Conv2d(in_channels, base_channels, 3, padding=1, dilation=1),    # 3x3
            nn.Conv2d(in_channels, base_channels, 3, padding=2, dilation=2),    # 5x5
            nn.Conv2d(in_channels, base_channels, 3, padding=4, dilation=4),    # 9x9  
            nn.Conv2d(in_channels, base_channels, 3, padding=8, dilation=8),    # 17x17
            nn.Conv2d(in_channels, base_channels, 3, padding=16, dilation=16),  # 33x33
        ])
        
    def forward(self, x):
        features = [scale(x) for scale in self.scales]
        return torch.cat(features, dim=1)  # Concat all scales
```

---

## Design Patterns

### Exponential Dilation Growth
```python
# Common pattern: exponentially increase dilation
dilations = [1, 2, 4, 8, 16, 32]  # Covers 1x1 to 65x65 receptive field

layers = []
for dilation in dilations:
    layers.append(
        nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
    )
```

### Hybrid Conv-Dilated Architecture
```python
class HybridBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Regular conv for local features
        self.local = nn.Conv2d(channels, channels//2, 3, padding=1)
        
        # Dilated conv for global context
        self.global_ctx = nn.Conv2d(channels, channels//2, 3, padding=4, dilation=4)
        
    def forward(self, x):
        local_feat = self.local(x)
        global_feat = self.global_ctx(x)
        return torch.cat([local_feat, global_feat], dim=1)
```

### Gradual Dilation Increase
```python
# Smooth transition from fine to coarse
class GradualDilation(nn.Module):
    def __init__(self, channels, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return x
```

---

## Receptive Field Calculation

### Formula
```python
def receptive_field_size(kernel_size, dilation):
    """Calculate effective receptive field."""
    return (kernel_size - 1) * dilation + 1

# Examples
print(receptive_field_size(3, 1))   # 3 (regular 3x3)
print(receptive_field_size(3, 2))   # 5 
print(receptive_field_size(3, 4))   # 9
print(receptive_field_size(3, 8))   # 17
print(receptive_field_size(3, 16))  # 33
```

### Stacked Dilated Convolutions
```python
def stacked_receptive_field(kernel_sizes, dilations):
    """Calculate receptive field of stacked dilated convs."""
    total_rf = 1
    for k, d in zip(kernel_sizes, dilations):
        layer_rf = (k - 1) * d
        total_rf += layer_rf
    return total_rf

# WaveNet example
dilations = [1, 2, 4, 8, 16, 32, 64, 128]  # 8 layers
rfs = [stacked_receptive_field([2]*i, dilations[:i]) for i in range(1, len(dilations)+1)]
print(rfs)  # Shows growing receptive field
```

---

## Training Tips

### Memory Optimization
```python
# Dilated convolutions can be memory-intensive
# Use gradient checkpointing for very large receptive fields
model = torch.utils.checkpoint.checkpoint_sequential(dilated_layers, 2, x)

# Or use separable dilated convolutions
class SeparableDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Depthwise dilated conv
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  padding=dilation*(kernel_size-1)//2, 
                                  dilation=dilation, groups=in_channels)
        # Pointwise conv
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

### Avoiding Gridding Artifacts
```python
# Problem: Regular dilation patterns can create gridding
# Solution: Use different dilation rates or random dilation

class AntiGriddingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Mix different dilation rates to avoid regular patterns
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=3, dilation=3)  # Different rate
        
    def forward(self, x):
        # Combine irregular dilations
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return (out1 + out2) / 2  # Average to reduce artifacts
```

---

## Performance Optimizations

### Efficient Implementation
```python
# Use grouped convolutions with dilation
class EfficientDilatedConv(nn.Module):
    def __init__(self, channels, dilation, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 
                             padding=dilation, dilation=dilation, 
                             groups=groups)  # Reduce computation
        
# Use 1D convolutions where possible (e.g., WaveNet)
class Efficient1DDilated(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, 
                             padding=dilation, dilation=dilation)
```

### Mixed Precision Training
```python
# Dilated convs work well with mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = dilated_model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Common Pitfalls & Solutions

### Issue: Checkerboard Artifacts
```python
# Problem: Regular dilation creates checkerboard patterns
# Solution: Use irregular dilation or post-processing

# Bad: Regular pattern
dilations = [2, 2, 2, 2]  # Creates artifacts

# Good: Irregular pattern  
dilations = [2, 3, 2, 3]  # Breaks regularity

# Or use anti-aliasing
class AntiAliasedDilated(nn.Module):
    def forward(self, x):
        # Apply slight blur after dilated conv
        out = self.dilated_conv(x)
        return F.avg_pool2d(out, 3, stride=1, padding=1)
```

### Issue: Parameter Explosion
```python
# Problem: Many parallel dilated convs = many parameters
# Solution: Use bottleneck design

class EfficientASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super().__init__()
        # Reduce channels first
        self.reduce = nn.Conv2d(in_channels, out_channels//4, 1)
        
        # Dilated convs on reduced channels
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels//4, out_channels//4, 3, 
                     padding=d, dilation=d) for d in dilations
        ])
        
        # Expand back
        self.expand = nn.Conv2d(len(dilations) * out_channels//4, out_channels, 1)
```

---

## Modern Applications

### Real-Time Segmentation
```python
# Use dilated convs in mobile networks
class MobileDilatedSeg(nn.Module):
    def __init__(self):
        super().__init__()
        # Efficient backbone
        self.backbone = mobilenet_v3_small()
        
        # Lightweight ASPP
        self.aspp = EfficientASPP(960, 128, [3, 6, 9])
        self.head = nn.Conv2d(128, 19, 1)  # PASCAL VOC classes
```

### Audio Processing
```python
# WaveNet for speech synthesis
class WaveNetStack(nn.Module):
    def __init__(self, channels=256, layers=30):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        dilation = 1
        for i in range(layers):
            self.blocks.append(WaveNetBlock(channels, dilation))
            dilation *= 2  # Exponential growth
            if dilation > 512:  # Reset cycle
                dilation = 1
```

---

*Dilated convolutions prove that sometimes the best way forward is to look at the bigger picture! 🔍🌍*