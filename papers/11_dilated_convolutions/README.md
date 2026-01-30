# Day 11: Multi-Scale Context Aggregation by Dilated Convolutions

> *"Multi-Scale Context Aggregation by Dilated Convolutions"* - Fisher Yu, Vladlen Koltun (2015)

**📖 Original Paper:** https://arxiv.org/abs/1511.07122

**⏱️ Time to Complete:** 4-5 hours

**🎯 What You'll Learn:**
- How to expand receptive fields without losing resolution
- The magic of dilated (atrous) convolutions
- Why pooling destroys information for dense prediction tasks
- How to capture multi-scale context efficiently
- The architecture behind WaveNet and DeepLab

---

## 🧠 The Big Idea

**In one sentence:** Dilated convolutions add **holes (zeros) between kernel weights** to exponentially expand receptive fields without losing spatial resolution or adding parameters.

### The Dense Prediction Problem

Tasks like segmentation, depth estimation, and image generation need:
- ✅ **Large receptive fields** - See the whole context
- ✅ **High resolution** - Preserve spatial details
- ❌ **No pooling** - Pooling destroys localization!

**But standard convolutions have a problem:**

```
Standard 3×3 conv:
- Receptive field: 3×3
- Stack 10 layers: 21×21 receptive field
- Need 100+ layers for full image context!
```

**Traditional solution (pooling):**
```
Conv → Pool → Conv → Pool → ... → Upsample
         ↓          ↓
    Lose resolution! Lose precise locations!
```

### The Dilated Solution

**Dilated convolutions:** Insert spaces (dilation) between kernel weights:

```
Rate 1 (standard):    Rate 2 (dilated):      Rate 4:
[x x x]               [x . x . x]            [x . . . x . . . x]
[x x x]               [. . . . .]            [. . . . . . . . .]
[x x x]               [x . x . x]            [. . . . . . . . .]
                      [. . . . .]            [. . . . . . . . .]
                      [x . x . x]            [x . . . x . . . x]

3×3 receptive field   7×7 receptive field!   15×15 receptive field!
9 parameters          9 parameters           9 parameters
```

**Result:** Exponential receptive field growth with NO additional parameters!

---

## 🤔 Why Dilated Convolutions are Brilliant

### The Math Behind Dilation

**Standard convolution:**
$$y[i] = \sum_{k=1}^{K} w[k] \cdot x[i + k]$$

**Dilated convolution (dilation rate $r$):**
$$y[i] = \sum_{k=1}^{K} w[k] \cdot x[i + r \cdot k]$$

**The $r$ parameter spaces out the kernel!**

### Receptive Field Growth

**Standard convolutions:**
- Layer 1: $3 \times 3 = 9$ receptive field
- Layer 2: $5 \times 5 = 25$
- Layer 3: $7 \times 7 = 49$
- Growth: **Linear** ($2n + 1$)

**Dilated convolutions (rate $2^{l}$):**
- Layer 1 (r=1): $3 \times 3 = 9$
- Layer 2 (r=2): $7 \times 7 = 49$
- Layer 3 (r=4): $15 \times 15 = 225$
- Layer 4 (r=8): $31 \times 31 = 961$
- Growth: **Exponential** ($2^{l+2} - 1$)

**With just 4 layers, we see 961 pixels!**

### Why Not Just Use Pooling?

**Pooling approach:**
```
256×256 → Pool → 128×128 → Pool → 64×64 → ... → Upsample → 256×256
           ❌           ❌           ❌              ❌
        Lost info    Lost info    Lost info    Blurry recovery
```

**Dilated approach:**
```
256×256 → Dilated → 256×256 → Dilated → 256×256
           ✅           ✅            ✅
     Full resolution maintained throughout!
```

**Result:** Perfect for segmentation, depth estimation, super-resolution!

---

## 🌍 Real-World Analogy

### The Telescope Analogy

**Standard Convolution (3×3):**
```
Looking through a magnifying glass
▓▓▓
▓○▓  ← Can only see immediate neighbors
▓▓▓
```
Small field of view, high detail

**Pooling:**
```
Telescope with lower resolution
████████████
████████████  ← See far away but blurry!
████████████
████████████
```
Large field of view, but details lost

**Dilated Convolution:**
```
Telescope with selective focus points
○ . . ○ . . ○
. . . . . . .
. . . . . . .
○ . . ○ . . ○  ← Sample distant points clearly!
. . . . . . .
. . . . . . .
○ . . ○ . . ○
```
Large field of view, maintains resolution!

### The Fishnet Analogy

**Standard Conv (dense weave):**
```
████████
████████  ← Catches small fish, limited area
████████
```

**Dilated Conv (wide-spread net):**
```
█ . . █ . . █
. . . . . . .
. . . . . . .
█ . . █ . . █  ← Covers huge area, same material!
```

Same amount of "rope" (parameters), but covers much more ocean (receptive field)!

---

## 📊 The Architecture

### Basic Dilated Convolution

```python
import torch
import torch.nn as nn

# Standard convolution
standard_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

# Dilated convolution (rate 2)
dilated_conv = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

# Dilated convolution (rate 4)
dilated_conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
```

**Key parameter:** `dilation` controls the spacing!

### Context Module Architecture

**The paper's context aggregation module:**

```
Input Feature Map (256×256×C)
        ↓
    Conv 3×3, dilation=1
        ↓
    Conv 3×3, dilation=1
        ↓
    Conv 3×3, dilation=2
        ↓
    Conv 3×3, dilation=4
        ↓
    Conv 3×3, dilation=8
        ↓
    Conv 3×3, dilation=16
        ↓
    Conv 3×3, dilation=32
        ↓
    Conv 3×3, dilation=64
        ↓
    Conv 1×1 (reduce channels)
        ↓
Output (256×256×C)

Receptive field: ENTIRE IMAGE!
Resolution: UNCHANGED!
```

### Front-End Module

Removes pooling from VGG/ResNet:

```python
class DilatedFrontEnd(nn.Module):
    """Convert VGG/ResNet to dilated version"""
    
    def __init__(self, pretrained_model):
        super().__init__()
        
        # Keep early layers (with pooling)
        self.layer1 = pretrained_model.layer1  # stride 1
        self.layer2 = pretrained_model.layer2  # stride 2
        self.layer3 = pretrained_model.layer3  # stride 2
        
        # Remove pooling, add dilation
        self.layer4 = self._convert_to_dilated(
            pretrained_model.layer4,
            dilation=2,
            stride=1  # Remove stride!
        )
        self.layer5 = self._convert_to_dilated(
            pretrained_model.layer5,
            dilation=4,
            stride=1  # Remove stride!
        )
    
    def _convert_to_dilated(self, layer, dilation, stride):
        """Convert strided layer to dilated"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                # Replace stride with dilation
                if module.stride == (2, 2):
                    module.stride = (stride, stride)
                    module.dilation = (dilation, dilation)
                    # Adjust padding for dilation
                    module.padding = (dilation, dilation)
        return layer
```

---

## 🔧 Implementation Guide

### Dilated Convolution Block

```python
class DilatedConvBlock(nn.Module):
    """Basic dilated convolution block"""
    
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        
        # Padding = dilation for same output size
        padding = dilation
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Example usage
block1 = DilatedConvBlock(64, 64, dilation=1)   # 3×3 receptive field
block2 = DilatedConvBlock(64, 64, dilation=2)   # 7×7 receptive field  
block4 = DilatedConvBlock(64, 64, dilation=4)   # 15×15 receptive field
block8 = DilatedConvBlock(64, 64, dilation=8)   # 31×31 receptive field
```

### Multi-Scale Context Module

```python
class ContextModule(nn.Module):
    """Multi-scale context aggregation"""
    
    def __init__(self, channels):
        super().__init__()
        
        # Exponentially increasing dilation rates
        self.dilations = [1, 1, 2, 4, 8, 16, 32, 64]
        
        # Build layers
        layers = []
        for dilation in self.dilations:
            layers.append(
                DilatedConvBlock(channels, channels, dilation)
            )
        self.layers = nn.ModuleList(layers)
        
        # Final 1×1 to combine
        self.final_conv = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        # Sequential application
        for layer in self.layers:
            x = layer(x)
        
        # Final combination
        x = self.final_conv(x)
        return x

# Usage
context = ContextModule(512)
output = context(features)  # Same shape as input!
```

### ASPP (Atrous Spatial Pyramid Pooling)

**DeepLab's improvement:** Parallel dilated convs at multiple scales

```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLab)"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple parallel dilated convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Batch norms
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # Combine branches
        self.conv_cat = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_cat = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Parallel branches
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        
        # Global pooling branch
        x5 = self.global_pool(x)
        x5 = nn.functional.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate and combine
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn_cat(self.conv_cat(x)))
        
        return x
```

### Full Segmentation Network

```python
class DilatedSegNet(nn.Module):
    """Complete segmentation network with dilated convolutions"""
    
    def __init__(self, num_classes=21):
        super().__init__()
        
        # Encoder (could be ResNet, VGG, etc.)
        self.encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            # Conv2 - standard
            self._make_layer(64, 64, 3, stride=1, dilation=1),
            
            # Conv3 - standard
            self._make_layer(64, 128, 4, stride=2, dilation=1),
            
            # Conv4 - dilated (remove stride, add dilation)
            self._make_layer(128, 256, 6, stride=1, dilation=2),
            
            # Conv5 - dilated (remove stride, add dilation)
            self._make_layer(256, 512, 3, stride=1, dilation=4),
        )
        
        # Context module
        self.context = ContextModule(512)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride, dilation):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(DilatedConvBlock(in_ch, out_ch, dilation))
            else:
                layers.append(DilatedConvBlock(out_ch, out_ch, dilation))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Encode
        x = self.encoder(x)
        
        # Aggregate context
        x = self.context(x)
        
        # Classify
        x = self.classifier(x)
        
        # Upsample to input size
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x
```

---

## 🎯 Training Tips

### 1. **Output Stride Matters**

Control how much downsampling happens:

```python
def set_output_stride(model, output_stride=8):
    """
    output_stride = 32: Standard (5× downsampling)
    output_stride = 16: Less downsampling (better)
    output_stride = 8:  Minimal downsampling (best, slower)
    """
    
    if output_stride == 16:
        # Layer4: stride=1, dilation=2
        model.layer4.apply(lambda m: setattr(m, 'dilation', 2) if hasattr(m, 'dilation') else None)
    elif output_stride == 8:
        # Layer3: stride=1, dilation=2
        # Layer4: stride=1, dilation=4
        model.layer3.apply(lambda m: setattr(m, 'dilation', 2) if hasattr(m, 'dilation') else None)
        model.layer4.apply(lambda m: setattr(m, 'dilation', 4) if hasattr(m, 'dilation') else None)
```

### 2. **Multi-Scale Training**

Train with different input sizes:

```python
def multi_scale_training():
    """Train with random scales"""
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    for epoch in range(epochs):
        for images, labels in dataloader:
            # Random scale
            scale = random.choice(scales)
            h, w = images.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear')
            labels = F.interpolate(labels.float(), size=(new_h, new_w), mode='nearest').long()
            
            # Train
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

### 3. **Learning Rate Policy**

Poly learning rate schedule works well:

```python
def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    """Polynomial learning rate decay"""
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

### 4. **Loss Functions**

For segmentation, use weighted cross-entropy:

```python
def get_class_weights(dataloader, num_classes):
    """Calculate inverse frequency weights"""
    class_counts = torch.zeros(num_classes)
    
    for _, labels in dataloader:
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()
    
    # Inverse frequency
    weights = 1.0 / (class_counts + 1)
    weights = weights / weights.sum() * num_classes
    
    return weights

# Usage
weights = get_class_weights(train_loader, num_classes=21)
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 📈 Visualizations

### 1. Receptive Field Visualization

```python
def visualize_receptive_field():
    """Show receptive field growth"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    layers = np.arange(1, 11)
    
    # Standard convolution
    rf_standard = 2 * layers + 1
    
    # Dilated convolution (dilation = 2^l)
    rf_dilated = [3]  # First layer
    for l in range(1, 10):
        dilation = 2 ** l
        rf_dilated.append(rf_dilated[-1] + 2 * dilation)
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, rf_standard, 'o-', label='Standard Conv', linewidth=2, markersize=8)
    plt.plot(layers, rf_dilated, 's-', label='Dilated Conv (rate 2^l)', linewidth=2, markersize=8)
    plt.xlabel('Layer Number', fontsize=12)
    plt.ylabel('Receptive Field Size', fontsize=12)
    plt.title('Dilated Convolutions: Exponential Receptive Field Growth!', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

visualize_receptive_field()
```

### 2. Dilation Pattern Visualization

```python
def show_dilation_patterns():
    """Visualize different dilation rates"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, dilation in enumerate([1, 2, 4, 8]):
        # Create grid
        size = 3 + (3 - 1) * (dilation - 1)
        grid = np.zeros((size, size))
        
        # Mark kernel positions
        for i in range(3):
            for j in range(3):
                grid[i * dilation, j * dilation] = 1
        
        # Plot
        axes[idx].imshow(grid, cmap='RdBu_r', vmin=0, vmax=1)
        axes[idx].set_title(f'Dilation Rate = {dilation}\nReceptive Field: {size}×{size}', fontsize=12)
        axes[idx].axis('off')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('3×3 Kernel with Different Dilation Rates', fontsize=14)
    plt.tight_layout()
    plt.show()
```

### 3. Feature Map Resolution

```python
def compare_architectures():
    """Compare resolution preservation"""
    
    architectures = ['FCN\n(with pooling)', 'Dilated\n(no pooling)']
    resolutions = [
        [224, 112, 56, 28, 14, 7, 14, 28, 56, 112, 224],  # FCN: down then up
        [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224]  # Dilated: constant
    ]
    
    plt.figure(figsize=(14, 6))
    
    for idx, (arch, res) in enumerate(zip(architectures, resolutions)):
        plt.subplot(1, 2, idx + 1)
        plt.plot(res, 'o-', linewidth=3, markersize=10)
        plt.axhline(y=224, color='r', linestyle='--', label='Input resolution')
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Feature Map Size', fontsize=12)
        plt.title(arch, fontsize=14)
        plt.ylim(0, 240)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
```

---

## 🏋️ Exercises

### Exercise 1: Receptive Field Exploration (⏱️⏱️)
Implement dilated convolutions with different rates. Visualize the effective receptive field for each configuration. Compare:
- Rate [1, 1, 1, 1]: Standard stacking
- Rate [1, 2, 4, 8]: Exponential dilation
- Rate [2, 2, 2, 2]: Constant dilation

### Exercise 2: Build Context Module (⏱️⏱️⏱️)
Implement the full context aggregation module from the paper. Test on CIFAR-10 segmentation:
- Sequential dilated convs (1, 1, 2, 4, 8, 16, 32, 64)
- Measure receptive field at each layer
- Compare with standard convolutions

### Exercise 3: ASPP Implementation (⏱️⏱️⏱️⏱️)
Build Atrous Spatial Pyramid Pooling:
- Parallel branches with different dilations
- Global pooling branch
- Concatenation and fusion
- Test on Pascal VOC segmentation

### Exercise 4: Ablation Study (⏱️⏱️⏱️⏱️)
Compare different architectures on segmentation:
1. FCN (with pooling + upsampling)
2. Dilated network (no pooling)
3. Dilated + ASPP
4. Dilated + Multi-scale training

Which combination works best?

### Exercise 5: WaveNet Audio Generation (⏱️⏱️⏱️⏱️⏱️)
Implement WaveNet using dilated causal convolutions:
- Causal (1D) dilated convolutions
- Gated activation units
- Residual connections
- Train on audio waveforms

Can you generate realistic audio?

---

## 🚀 Going Further

### Applications of Dilated Convolutions

**Computer Vision:**
- ✅ Semantic segmentation (DeepLab, PSPNet)
- ✅ Instance segmentation (Mask R-CNN)
- ✅ Depth estimation
- ✅ Super-resolution
- ✅ Video processing

**Audio:**
- ✅ WaveNet (speech synthesis)
- ✅ Audio classification
- ✅ Music generation

**Other Domains:**
- ✅ Time series prediction
- ✅ Video frame prediction
- ✅ Point cloud processing

### Evolution of the Idea

**DeepLab V1 (2014):**
- Introduced dilated convolutions for segmentation
- ASPP for multi-scale context

**DeepLab V2 (2016):**
- Improved ASPP
- Multiple dilation rates

**DeepLab V3 (2017):**
- Removed CRF post-processing
- Better ASPP design

**DeepLab V3+ (2018):**
- Added encoder-decoder structure
- Atrous separable convolutions

### Modern Usage

**What's Still Used:**
- ✅ Dilated convolutions (everywhere!)
- ✅ ASPP (multi-scale aggregation)
- ✅ Output stride control

**What's Been Improved:**
- Separable dilated convolutions (fewer parameters)
- Self-attention mechanisms
- Transformer-based segmentation

---

## 📚 Resources

### Must-Read
- 📄 [Original Paper](https://arxiv.org/abs/1511.07122) - Dilated Convolutions
- 📄 [DeepLab V3+](https://arxiv.org/abs/1802.02611) - Modern usage
- 📄 [WaveNet](https://arxiv.org/abs/1609.03499) - Audio application

### Implementations
- 💻 [PyTorch DeepLab](https://github.com/jfzhang95/pytorch-deeplab-xception) - Full implementation
- 💻 [TensorFlow DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) - Official
- 💻 [WaveNet PyTorch](https://github.com/vincentherrmann/pytorch-wavenet) - Audio generation

### Visualizations
- 🎥 [Dilated Convolutions Explained](https://www.youtube.com/watch?v=HXy7X8H00U0) - Video
- 📊 [Interactive Demo](https://ezyang.github.io/convolution-visualizer/) - Visualize dilations

---

## 🎓 Key Takeaways

### 1. **The Core Insight**

Dilated convolutions solve a fundamental trade-off:
- **Need:** Large receptive field
- **Also need:** High resolution
- **Solution:** Insert spaces between kernel weights!

### 2. **The Math is Simple**

From standard convolution:
$$y[i] = \sum_{k} w[k] \cdot x[i + k]$$

To dilated convolution:
$$y[i] = \sum_{k} w[k] \cdot x[i + r \cdot k]$$

Just multiply the offset by dilation rate $r$!

### 3. **Exponential Growth**

**Standard:** Linear receptive field growth  
**Dilated:** Exponential receptive field growth

With dilation rates [1, 2, 4, 8], you see 31× the area with same parameters!

### 4. **Universal Pattern**

The multi-scale aggregation pattern is everywhere:

```python
# The universal pattern
x1 = conv_dilation_1(x)   # Fine details
x2 = conv_dilation_2(x)   # Medium context
x3 = conv_dilation_4(x)   # Large context
output = combine([x1, x2, x3])
```

### 5. **Trade-offs**

**Advantages:**
- ✅ Large receptive field
- ✅ High resolution
- ✅ No extra parameters
- ✅ Efficient

**Disadvantages:**
- ❌ More memory (high-res features)
- ❌ Gridding artifacts (if not careful)
- ❌ Slower than pooling

---

## 💡 Why This Matters Today

### The Dense Prediction Revolution

Dilated convolutions enabled modern dense prediction:

**Before (2015):**
```
Image → Encode (pool) → Decode (upsample) → Segmentation
                ❌ Lost resolution!
```

**After (2015+):**
```
Image → Dilated Network → Segmentation
            ✅ Maintains resolution!
```

### The Pattern Lives On

Even with Transformers taking over, the core insights remain:

**Vision Transformers:**
- Use dilated attention (shifted windows)
- Multi-scale feature pyramids
- Same principle: capture context without losing resolution!

### Practical Impact

**What changed after dilated convolutions:**
- Segmentation: 60% → 90% mIoU on Pascal VOC
- Real-time segmentation became possible
- Audio generation (WaveNet) achieved human-like speech
- Every modern segmentation network uses the ideas

---

## 🔍 Comparing Approaches

| Approach | Receptive Field | Resolution | Parameters | Speed |
|----------|----------------|------------|------------|-------|
| **Standard Conv** | Small (linear) | High | Low | Fast |
| **Pooling + Upsample** | Large | Low→High | Low | Fast |
| **Dilated Conv** | Large (exponential) | High | Low | Medium |
| **Self-Attention** | Global | High | High | Slow |

**Best choice depends on task:**
- Classification: Standard + pooling
- Segmentation: Dilated convolutions
- High-res generation: Dilated + attention

---

**Completed Day 11?** You've mastered multi-scale context aggregation! Time to explore modern architectures that build on these foundations.

**Questions?** Check the [notebook.ipynb](notebook.ipynb) for interactive visualizations and experiments.

---

*"The key idea is to use dilated convolutions to systematically aggregate multi-scale contextual information without losing resolution."* - Fisher Yu on the core insight of dilated convolutions