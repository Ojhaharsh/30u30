# Day 9: ResNet - Deep Residual Learning for Image Recognition

> *"Deep Residual Learning for Image Recognition"* - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)

**📖 Original Paper:** https://arxiv.org/abs/1512.03385

**⏱️ Time to Complete:** 4-5 hours

**🎯 What You'll Learn:**
- Why deeper networks performed WORSE (the degradation problem)
- How skip connections solve vanishing gradients
- The elegance of residual learning
- Why every modern architecture uses skip connections
- How to train 100+ layer networks successfully

---

## 🧠 The Big Idea

**In one sentence:** Skip connections (`output = F(x) + x`) create "gradient highways" that enable training networks 100+ layers deep by letting information and gradients flow directly through the network.

### The Paradox That Shouldn't Exist

Before ResNet, researchers discovered something deeply puzzling:

**Experiment:**
- Train a 20-layer network: 8.75% error
- Train a 56-layer network: 9.43% error (WORSE!)

**Wait, what?** 

The deeper network should at least be able to copy the shallow network's layers and add identity mappings. It should be **at least as good**, if not better!

But it wasn't. This was called the **degradation problem**.

### The ResNet Solution

**The insight:** Don't learn the desired mapping `H(x)` directly. Instead, learn the **residual** `F(x) = H(x) - x`, then add it back:

$$H(x) = F(x) + x$$

**Why this works:**
- If the optimal function is identity, just set `F(x) = 0` (easy!)
- Gradients flow backward through the `+x` path without attenuation
- Networks can now be 100+ layers deep and actually improve

**Result:** ResNet-152 won ImageNet 2015 with 3.57% error—beating human performance (5.1%)!

---

## 🤔 Why Skip Connections Changed Everything

This isn't just a trick—it's a **fundamental insight** about how deep networks learn:

**The core discovery:** Deep networks struggle not because of vanishing gradients alone, but because learning identity mappings is hard.

### The Degradation Problem

**What we thought:**
- Problem: Vanishing gradients
- Solution: Better initialization, normalization

**What ResNet discovered:**
- **Real problem**: Degradation during optimization
- Plain networks can't even learn identity mappings!
- Adding layers made things WORSE, even on training data

**The shocking result:**
```
Shallow network (20 layers): 8.75% training error
Deep network (56 layers): 9.43% training error

This isn't overfitting—it's UNDERFITTING!
```

### Why Residual Learning Works

**Learning Direct Mapping:**
```
Network must learn: H(x) = x (identity)
Difficult: Requires precise weight settings
```

**Learning Residual:**
```
Network must learn: F(x) = 0 (much easier!)
Then: H(x) = F(x) + x = 0 + x = x ✓
```

**Key insight:** It's easier to learn small adjustments to identity than to learn the entire mapping!

---

## 🌍 Real-World Analogy

### The River Analogy

Think of training as water flowing backward through the network:

**Plain Deep Network (No Skip Connections):**
```
Gradient Source → Layer 50 → Layer 49 → ... → Layer 2 → Layer 1
                   ↓         ↓               ↓         ↓
                  weak      weaker         vanishing  gone!
```

Like a river flowing through 50 waterfalls—by the end, there's barely a trickle!

**ResNet (With Skip Connections):**
```
Gradient Source ━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ Layer 1
                   ↓         ↓               ↓      (direct path!)
                Layer 50  Layer 49  ...  Layer 2
```

Like a river with a BYPASS canal! Water can flow directly through OR through layers. The gradient river never dries up!

### The Editing Analogy

**Without Skip Connections (Rewriting):**
- Start with blank page
- Write entire document from scratch
- Any mistake means starting over
- **Hard!**

**With Skip Connections (Editing):**
- Start with existing draft (`x`)
- Make small corrections (`F(x)`)
- Final version = draft + corrections
- Can leave parts unchanged
- **Easy!**

That's exactly how ResNet works—it edits the input rather than rewriting it!

---

## 📊 The Architecture

### The Residual Block

**Basic Building Block:**
```
           ┌────────────┐
    x ─────┤            ├───── H(x) = F(x) + x
       │   │  Conv-BN   │   │
       │   │    ReLU    │   │
       │   │  Conv-BN   │   │
       │   └────────────┘   │
       │         F(x)       │
       └──────────(+)───────┘
                  ↑
            Skip Connection
```

**Implementation:**
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path: F(x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection: identity or projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Save input for skip connection
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out
```

### ResNet-50 Architecture

```
Input: 224×224×3
     ↓
Conv1: 7×7×64, stride=2
     ↓
MaxPool: 3×3, stride=2
     ↓
Stage 1: [1×1×64, 3×3×64, 1×1×256] × 3 blocks
     ↓
Stage 2: [1×1×128, 3×3×128, 1×1×512] × 4 blocks  
     ↓
Stage 3: [1×1×256, 3×3×256, 1×1×1024] × 6 blocks
     ↓
Stage 4: [1×1×512, 3×3×512, 1×1×2048] × 3 blocks
     ↓
Global Average Pooling
     ↓
FC: 1000 classes
```

**Total depth:** 50 layers  
**Parameters:** ~25.6M

### Bottleneck Block (for deeper ResNets)

**Motivation:** Reduce computational cost for deep networks

```
           ┌─────────────────┐
    x ─────┤  1×1 Conv (↓64) ├───── H(x) = F(x) + x
       │   │  3×3 Conv (64)  │   │
       │   │  1×1 Conv (↑256)│   │
       │   └─────────────────┘   │
       └──────────(+)────────────┘
```

**Why it works:**
- 1×1 conv reduces dimensions (64 → 64 channels)
- 3×3 conv operates on fewer channels (faster!)
- 1×1 conv expands back to original dimensions
- Same receptive field, fewer FLOPs!

---

## 💡 The Vanishing Gradient Solution

### Why Plain Networks Fail

**Gradient flow in plain network:**
$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=1}^{n} \frac{\partial x_i}{\partial x_{i-1}}$$

Each multiplication can shrink gradients. After 50 layers: $(0.9)^{50} = 0.005$ (0.5%!)

### Why ResNet Succeeds

**Gradient flow through skip connection:**

$$H(x) = F(x) + x$$

$$\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + 1$$

**The magic "+1":** Gradient always has a direct path! Even if $\frac{\partial F}{\partial x} \approx 0$, the gradient flows through the identity.

**Backpropagation:**
```python
# Plain network
dL/dx = dL/dy * dy/dx  # Can vanish!

# ResNet
dL/dx = dL/dy * (dF/dx + 1)  # Always has the +1 term!
```

The gradient is **guaranteed** to flow backward, no matter how deep!

---

## 🔧 Implementation Guide

### Building ResNet from Scratch

```python
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    expansion = 4  # Output channels = in_channels * 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Bottleneck design: 1×1 → 3×3 → 1×1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (projection if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Add skip connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        
        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Create ResNet variants
def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes)
```

---

## 🎯 Training Tips

### 1. **Learning Rate Schedule**

ResNet uses learning rate warm-up and step decay:

```python
def get_lr(epoch, base_lr=0.1, warmup_epochs=5):
    """ResNet learning rate schedule"""
    if epoch < warmup_epochs:
        # Linear warm-up
        return base_lr * (epoch + 1) / warmup_epochs
    elif epoch < 30:
        return base_lr
    elif epoch < 60:
        return base_lr * 0.1
    elif epoch < 80:
        return base_lr * 0.01
    else:
        return base_lr * 0.001

# Apply schedule
for epoch in range(90):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### 2. **Weight Initialization (Kaiming Init)**

ResNet introduced **Kaiming Initialization** for ReLU networks:

```python
def kaiming_init(model):
    """Initialize ResNet weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

model = ResNet50()
kaiming_init(model)
```

### 3. **Batch Normalization is Critical**

Every conv layer should be followed by BN (except the last):

```python
# Standard ResNet pattern
conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)  # No bias!
bn = nn.BatchNorm2d(out_ch)
relu = nn.ReLU(inplace=True)

# Forward
out = relu(bn(conv(x)))
```

**Why bias=False?** BatchNorm has its own bias term, making conv bias redundant.

### 4. **Data Augmentation**

ResNet's augmentation strategy:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

---

## 📈 Visualizations

### 1. Gradient Flow Visualization

```python
def visualize_gradient_flow(model, named_parameters):
    """Plot gradient flow through network"""
    import matplotlib.pyplot as plt
    
    ave_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    
    plt.figure(figsize=(16, 6))
    plt.plot(ave_grads, alpha=0.7, color='b', linewidth=2)
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=2, color='r', linestyle='--')
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation='vertical')
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel('Layers')
    plt.ylabel('Average Gradient')
    plt.title('Gradient Flow Through ResNet')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage after backward()
visualize_gradient_flow(model, model.named_parameters())
```

### 2. Skip Connection Impact

```python
def compare_with_without_skip():
    """Compare ResNet with plain network"""
    
    # Plain network (no skip connections)
    class PlainBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out)  # No skip connection!
    
    # Train both and compare
    plain_net = nn.Sequential(*[PlainBlock(64) for _ in range(50)])
    resnet = nn.Sequential(*[BasicBlock(64, 64) for _ in range(50)])
    
    # ... training code ...
    
    plt.figure(figsize=(12, 5))
    plt.plot(plain_losses, label='Plain Network', linewidth=2)
    plt.plot(resnet_losses, label='ResNet', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Skip Connections Make Deep Networks Trainable!')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 🏋️ Exercises

### Exercise 1: Build ResNet-18 (⏱️⏱️⏱️)
Implement ResNet-18 from scratch. Train on CIFAR-10. Compare with a plain 18-layer network without skip connections.

### Exercise 2: Ablation Study (⏱️⏱️⏱️⏱️)
Test the impact of skip connections:
- Train ResNet-50 normally
- Remove skip connections (make it plain)
- Replace skip connections with concatenation (DenseNet-style)
- Try different skip patterns (every 2 layers, every 4 layers)

Which works best? Why?

### Exercise 3: Gradient Flow Analysis (⏱️⏱️⏱️)
Measure and visualize gradient magnitudes at different depths:
- Plain network: Watch gradients vanish
- ResNet: Observe healthy gradient flow
- Plot gradient norms vs layer depth

### Exercise 4: Depth Scaling (⏱️⏱️⏱️⏱️)
Train ResNets of increasing depth:
- ResNet-18, 34, 50, 101
- Plot accuracy vs depth
- Find the point of diminishing returns

### Exercise 5: Transfer Learning (⏱️⏱️⏱️)
Use pretrained ResNet-50 for a new task:
1. Load ImageNet weights
2. Replace final layer
3. Fine-tune on new dataset
4. Compare with training from scratch

---

## 🚀 Going Further

### ResNet Variants

**ResNeXt (2017):**
- Grouped convolutions
- "Cardinality" as new dimension
- Better accuracy with same complexity

**SE-ResNet (2018):**
- Squeeze-and-Excitation blocks
- Channel attention
- Adaptive recalibration

**ResNeSt (2020):**
- Split-attention
- Multi-path ResNet
- State-of-art on ImageNet

### Modern Improvements

**What's Still Used:**
- ✅ Skip connections (universal!)
- ✅ Bottleneck blocks
- ✅ Batch normalization

**What's Been Improved:**
- Pre-activation (ResNet-v2)
- Better downsampling (anti-aliasing)
- Attention mechanisms (CBAM, SE)
- Neural architecture search

---

## 📚 Resources

### Must-Read
- 📄 [Original Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning
- 📄 [Identity Mappings](https://arxiv.org/abs/1603.05027) - ResNet v2 (pre-activation)
- 📖 [Kaiming He's blog](http://kaiminghe.com/) - The creator of ResNet

### Implementations
- 💻 [PyTorch ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) - Official
- 💻 [TensorFlow ResNet](https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet) - TF official

### Visualizations
- 🎥 [ResNet Explained](https://www.youtube.com/watch?v=ZILIbUvp5lk) - Video tutorial
- 📊 [Distill: Building Blocks](https://distill.pub/2017/momentum/) - Understanding residuals

---

## 🎓 Key Takeaways

1. **Skip connections solve degradation** - Not just vanishing gradients!
2. **Residual learning is easier** - Learn F(x) = 0 instead of H(x) = x
3. **Gradients always flow** - The "+1" term guarantees it
4. **Depth now helps** - Deeper networks are actually better!
5. **Universal architecture** - Every modern network uses skip connections

### Why ResNet Changed Everything

ResNet didn't just win ImageNet—it **changed how we think about deep learning**:

**Before ResNet:** "We can't train very deep networks"  
**After ResNet:** "How deep can we go?"

Skip connections are now **everywhere:**
- Vision Transformers
- BERT, GPT (residuals in every layer!)
- U-Net (medical imaging)
- Diffusion models
- Graph neural networks

### The Philosophical Insight

ResNet teaches us: **Sometimes the best way forward is to explicitly preserve the past.**

Don't try to learn everything from scratch—build on what you already have!

---

**Completed Day 9?** Move on to **[Day 10: ResNet V2](../day10_resnet_v2/)** where pre-activation makes residual learning even better!

**Questions?** Check the [notebook.ipynb](notebook.ipynb) for interactive explorations.

---

*"The highway networks paper was published before our work. The network depth they achieved was not so high because of optimization difficulties... Our key innovation was to identify that the degradation problem is, at its core, an optimization problem."* - Kaiming He on ResNet's insight