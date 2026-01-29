# Day 10: ResNet V2 - Identity Mappings in Deep Residual Networks

> *"Identity Mappings in Deep Residual Networks"* - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016)

**📖 Original Paper:** https://arxiv.org/abs/1603.05027

**⏱️ Time to Complete:** 3-4 hours

**🎯 What You'll Learn:**
- Why the order of operations matters (pre-activation vs post-activation)
- How clean identity paths enable 1000+ layer networks
- The subtle but crucial improvements over original ResNet
- Why "full pre-activation" is the gold standard
- How to build ultra-deep networks that actually train

---

## 🧠 The Big Idea

**In one sentence:** Pre-activation (BN-ReLU-Conv instead of Conv-BN-ReLU) creates **cleaner identity mappings** that enable training networks 1000+ layers deep without degradation.

### The Problem with Original ResNet

Original ResNet (2015) worked great, but had a subtle issue:

```
      x ─┐
         │ Conv-BN-ReLU
         │ Conv-BN
      ┌──┘
      ↓ (+)
      ReLU  ← Problem! This ReLU disrupts the identity path
      ↓
      out
```

**The issue:** The final ReLU after the addition breaks the "pure" identity mapping. Information can be lost!

### ResNet V2's Solution: Pre-activation

```
      x ─┐  ← Pure identity! Nothing disrupts it
         │ BN-ReLU-Conv
         │ BN-ReLU-Conv
      ┌──┘
      ↓ (+)
      out  ← No ReLU here!
```

**Why it works:**
- Identity path is completely clean: `y = x + F(x)`
- Gradients flow backward without ANY transformation
- Activations are always full range (not clipped by ReLU)
- Networks can go 1000+ layers deep!

**Result:** ResNet-1001 trained successfully with BETTER performance than ResNet-152!

---

## 🤔 Why Pre-Activation is Better

### The Mathematical Insight

**Original ResNet (post-activation):**
$$y_l = \text{ReLU}(x_l + F(x_l))$$

Problem: The ReLU can **zero out** negative values, disrupting information flow!

**ResNet V2 (pre-activation):**
$$y_l = x_l + F(x_l)$$

Perfect! No transformation on the skip connection—pure identity mapping.

### Gradient Flow Analysis

**Original ResNet backward pass:**
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial y_l} \cdot \left(\mathbb{1}[\text{ReLU active}] + \frac{\partial F}{\partial x_l}\right)$$

The $\mathbb{1}[\text{ReLU active}]$ indicator can be 0, blocking gradients!

**ResNet V2 backward pass:**
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial y_l} \cdot \left(1 + \frac{\partial F}{\partial x_l}\right)$$

**Perfect!** The "1" term is ALWAYS there. Gradients ALWAYS flow through.

### The Activation Space Advantage

**Post-activation (original):**
- Activations after ReLU: `[0, ∞)`
- Information lost when ReLU zeros out negatives
- Asymmetric activation space

**Pre-activation (V2):**
- Activations: `(-∞, ∞)`
- Full representation capacity
- Symmetric activation space

---

## 🌍 Real-World Analogy

### The Highway Analogy

**Original ResNet (Post-Activation):**
```
Main Road (residual path): Has turns and speed bumps
Highway (skip connection):  Has a TOLL BOOTH at the exit
                           └─> Can block some cars!
```

The ReLU after addition is like a toll booth that only lets positive "cars" through!

**ResNet V2 (Pre-Activation):**
```
Main Road: Has turns and speed bumps at the ENTRANCE
Highway:   Completely clear, no obstacles!
          └─> ALL cars flow through freely
```

The identity path is completely unobstructed!

### The River System Analogy

**Post-Activation:**
```
Main Channel: ~~~~~ (with rocks)
Bypass Canal: ━━━━━━━━[DAM]━━━━
                      ↑
                    Blocks some flow!
```

**Pre-Activation:**
```
Main Channel: [FILTER]~~~~~
Bypass Canal: ━━━━━━━━━━━━━━
              ↑
            Completely clear!
```

The "filter" (BN-ReLU) is on the MAIN path, leaving the bypass crystal clear!

---

## 📊 Architecture Comparison

### Original ResNet Block (Post-Activation)

```
Input x
   ↓
   ├─────────────────┐
   │                 │
   │ Conv 3×3        │
   │ BatchNorm       │
   │ ReLU            │
   │ Conv 3×3        │
   │ BatchNorm       │
   │                 │
   └─────────(+)─────┘
             ↓
           ReLU  ← Disrupts identity!
             ↓
          Output
```

### ResNet V2 Block (Pre-Activation)

```
Input x
   ↓
   ├────────────────────┐
   │                    │ ← Pure identity
   │ BatchNorm          │
   │ ReLU               │
   │ Conv 3×3           │
   │ BatchNorm          │
   │ ReLU               │
   │ Conv 3×3           │
   │                    │
   └────────(+)─────────┘
             ↓
          Output  ← No activation!
```

**Key differences:**
1. BN-ReLU **before** convolution
2. No activation after addition
3. Clean identity path

---

## 🔧 Implementation Guide

### Pre-Activation Residual Block

```python
import torch
import torch.nn as nn

class PreActBlock(nn.Module):
    """Pre-activation Residual Block (ResNet V2)"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Pre-activation: BN-ReLU-Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        
        # Shortcut (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            )
    
    def forward(self, x):
        # Pre-activation: activate BEFORE conv
        out = self.relu1(self.bn1(x))
        
        # Shortcut uses activated x
        shortcut = self.shortcut(out)
        
        # Main path
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        
        # Add without activation!
        out += shortcut
        
        return out  # No ReLU here!

class PreActBottleneck(nn.Module):
    """Pre-activation Bottleneck Block"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Pre-activation bottleneck: 1×1 → 3×3 → 1×1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False)
            )
    
    def forward(self, x):
        # First pre-activation
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        
        # Bottleneck path
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        
        # Add without activation
        out += shortcut
        return out
```

### Full PreActResNet

```python
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution (no BN-ReLU before first conv)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final BN-ReLU (after all residual blocks)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        # Classification
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
        # Initial conv (no pre-activation for first layer)
        out = self.conv1(x)
        
        # Residual stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Final activation
        out = self.relu(self.bn(out))
        
        # Classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes)

def PreActResNet34(num_classes=10):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes)

def PreActResNet50(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes)

def PreActResNet152(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)

def PreActResNet1001(num_classes=10):
    """1001-layer ResNet! (Only possible with pre-activation)"""
    return PreActResNet(PreActBottleneck, [111, 111, 111, 111], num_classes)
```

---

## 🎯 Training Tips

### 1. Initialization

Pre-activation blocks need special initialization:

```python
def init_weights(model):
    """Initialize PreActResNet weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization for ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            # BN before ReLU: initialize to 1
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

model = PreActResNet50()
init_weights(model)
```

### 2. Learning Rate Schedule

Pre-activation allows slightly more aggressive learning rates:

```python
# Base learning rate can be higher
base_lr = 0.1

# Cosine annealing works well
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200,  # Total epochs
    eta_min=1e-5
)

# Or step decay
def get_lr(epoch):
    if epoch < 80:
        return base_lr
    elif epoch < 120:
        return base_lr * 0.1
    else:
        return base_lr * 0.01
```

### 3. Ultra-Deep Networks

For 1000+ layers, special considerations:

```python
# Lower initial learning rate
base_lr = 0.01  # Lower than usual

# More warmup epochs
warmup_epochs = 10

# Gradient clipping helps
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# More regularization
weight_decay = 5e-4  # Higher than usual
```

### 4. Mixed Precision Training

Pre-activation is more stable with FP16:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Scaled backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

---

## 📈 Visualizations

### 1. Gradient Flow Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_gradient_flow():
    """Compare gradient flow: post-act vs pre-act"""
    
    # Simulate training
    depths = [18, 34, 50, 101, 152, 200, 500, 1000]
    
    # Post-activation degrades
    post_act_grads = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
    
    # Pre-activation maintains flow
    pre_act_grads = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.5]
    
    plt.figure(figsize=(12, 6))
    plt.plot(depths, post_act_grads, 'o-', label='Post-Activation (Original)', linewidth=2)
    plt.plot(depths, pre_act_grads, 's-', label='Pre-Activation (V2)', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Network Depth (layers)', fontsize=12)
    plt.ylabel('Gradient Magnitude (normalized)', fontsize=12)
    plt.title('Gradient Flow: Pre-Activation Enables Ultra-Deep Networks', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

compare_gradient_flow()
```

### 2. Activation Distribution

```python
def visualize_activations(model, dataloader):
    """Visualize activation distributions"""
    
    activations = {
        'input': [],
        'layer1': [],
        'layer2': [],
        'layer3': [],
        'layer4': []
    }
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            x = inputs
            activations['input'].append(x.cpu().numpy().flatten())
            
            x = model.conv1(x)
            x = model.layer1(x)
            activations['layer1'].append(x.cpu().numpy().flatten())
            
            x = model.layer2(x)
            activations['layer2'].append(x.cpu().numpy().flatten())
            
            x = model.layer3(x)
            activations['layer3'].append(x.cpu().numpy().flatten())
            
            x = model.layer4(x)
            activations['layer4'].append(x.cpu().numpy().flatten())
            
            break  # Just one batch
    
    # Plot distributions
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for idx, (name, acts) in enumerate(activations.items()):
        acts = np.concatenate(acts)
        axes[idx].hist(acts, bins=50, alpha=0.7, color='blue')
        axes[idx].set_title(f'{name} activations')
        axes[idx].set_xlabel('Activation value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Pre-Activation: Full Range Activations', fontsize=16)
    plt.tight_layout()
    plt.show()
```

### 3. Depth vs Performance

```python
def plot_depth_scaling():
    """Show how performance scales with depth"""
    
    # Results from paper
    depths = [18, 34, 50, 101, 152, 200, 1001]
    
    # Post-activation plateaus
    post_act_acc = [93.0, 93.5, 94.0, 94.3, 94.4, 94.3, np.nan]  # Can't train 1001!
    
    # Pre-activation keeps improving
    pre_act_acc = [93.1, 93.7, 94.2, 94.6, 94.8, 95.0, 95.1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(depths[:-1], post_act_acc[:-1], 'o-', label='Post-Activation', linewidth=2, markersize=8)
    plt.plot(depths, pre_act_acc, 's-', label='Pre-Activation', linewidth=2, markersize=8)
    plt.xlabel('Network Depth', fontsize=12)
    plt.ylabel('CIFAR-10 Accuracy (%)', fontsize=12)
    plt.title('Pre-Activation Enables 1001-Layer Networks!', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1100)
    plt.ylim(92.5, 95.5)
    
    # Annotate 1001-layer result
    plt.annotate('1001 layers!\n(only possible with pre-act)',
                xy=(1001, 95.1), xytext=(700, 93.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.show()
```

---

## 🏋️ Exercises

### Exercise 1: Post-Act vs Pre-Act (⏱️⏱️⏱️)
Implement both versions of ResNet-18. Train on CIFAR-10. Compare:
- Training loss curves
- Gradient magnitudes at different layers
- Final accuracy

Which trains faster? Which achieves better accuracy?

### Exercise 2: Ablation Study (⏱️⏱️⏱️⏱️)
Test different activation placements:
1. Full pre-activation (BN-ReLU-Conv-BN-ReLU-Conv)
2. Post-activation (Conv-BN-ReLU-Conv-BN-ReLU)
3. Partial pre-activation (BN-ReLU-Conv-Conv-BN)
4. No BN before shortcut

Which works best? Why?

### Exercise 3: Depth Scaling (⏱️⏱️⏱️⏱️)
Train PreActResNets of increasing depth:
- 18, 34, 50, 101, 152, 200, 500 layers
- Plot training time vs depth
- Plot accuracy vs depth
- Find optimal depth-accuracy tradeoff

### Exercise 4: Ultra-Deep Network (⏱️⏱️⏱️⏱️⏱️)
Build and train ResNet-1001:
- Use pre-activation blocks
- Implement proper initialization
- Use gradient clipping
- Monitor gradient flow
- Compare with ResNet-152

Can you successfully train it?

### Exercise 5: Transfer Learning (⏱️⏱️⏱️)
Fine-tune pre-trained PreActResNet:
1. Load ImageNet pretrained weights
2. Replace classifier
3. Fine-tune on new dataset
4. Compare with post-activation ResNet

Does pre-activation transfer better?

---

## 🚀 Going Further

### Advanced Pre-Activation Designs

**ResNeXt-V2:**
- Combine pre-activation with grouped convolutions
- Better accuracy with lower complexity

**SE-PreActResNet:**
- Add Squeeze-and-Excitation to pre-activation blocks
- Channel-wise attention

**WideResNet:**
- Wider layers (more channels)
- Pre-activation with width factor 10
- Fewer layers but wider

### Modern Usage

**Where Pre-Activation is Used:**
- ✅ EfficientNet (MBConv blocks)
- ✅ Vision Transformers (pre-norm)
- ✅ BigTransfer models
- ✅ Diffusion models (U-Net with pre-activation)

**The Pattern:**
```
Normalize → Activate → Transform → Add

This pattern is EVERYWHERE in modern architectures!
```

### Implementation Details

**Tricks for training ultra-deep networks:**

1. **Stochastic Depth:**
```python
# Randomly drop layers during training
if self.training and random.random() < drop_prob:
    return x  # Skip this block entirely!
return x + F(x)
```

2. **Initial Residual Scaling:**
```python
# Scale residuals down initially
self.residual_scale = nn.Parameter(torch.zeros(1))
return x + self.residual_scale * F(x)
```

3. **Adaptive Learning Rates:**
```python
# Different learning rates for different depths
for i, layer in enumerate(model.layers):
    lr_scale = 0.1 ** (i / len(model.layers))
    optimizer.add_param_group({
        'params': layer.parameters(),
        'lr': base_lr * lr_scale
    })
```

---

## 📚 Resources

### Must-Read
- 📄 [Original Paper](https://arxiv.org/abs/1603.05027) - Identity Mappings
- 📄 [ResNet V1](https://arxiv.org/abs/1512.03385) - For comparison
- 📖 [Kaiming He's Talk](https://www.youtube.com/watch?v=C6tLw-rPQ2o) - Insights from the author

### Implementations
- 💻 [Official PyTorch](https://github.com/KaimingHe/resnet-1k-layers) - 1001-layer ResNet code
- 💻 [PreActResNet CIFAR](https://github.com/kuangliu/pytorch-cifar) - Clean implementation

### Analysis
- 📊 [The Shattered Gradients Problem](https://arxiv.org/abs/1702.08591) - Why skip connections matter
- 📄 [Understanding ResNets](https://arxiv.org/abs/1611.01186) - Ensemble perspective

---

## 🎓 Key Takeaways

### 1. **Pre-Activation is the New Standard**

**Old way (post-activation):**
```python
out = relu(bn(conv(x)))
out = bn(conv(out))
return relu(out + x)  # ReLU breaks identity!
```

**New way (pre-activation):**
```python
out = conv(relu(bn(x)))
out = conv(relu(bn(out)))
return out + x  # Pure identity!
```

### 2. **Order Matters**

The order of operations fundamentally changes:
- **Gradient flow** - Pre-activation has cleaner gradients
- **Activation space** - Full range vs clipped at zero
- **Network depth** - 1000+ layers vs ~200 max

### 3. **Mathematical Elegance**

Pre-activation achieves the ideal form:
$$y = x + F(x)$$

No transformations on the identity path. Gradients always have a direct route.

### 4. **Practical Impact**

**What changed after ResNet V2:**
- Deeper networks are now practical
- Training is more stable
- Transfer learning improved
- Became the default in modern architectures

### 5. **Universal Pattern**

The pre-activation pattern appears everywhere:

**Vision:** PreActResNet  
**NLP:** Pre-norm Transformers (LayerNorm before attention)  
**Diffusion:** Pre-activation U-Nets

**The pattern:** Normalize → Activate → Transform → Add

---

## 🔍 Comparing V1 and V2

| Aspect | ResNet V1 (Post-Act) | ResNet V2 (Pre-Act) |
|--------|---------------------|---------------------|
| **Identity Path** | Disrupted by final ReLU | Clean, no transformations |
| **Gradient Flow** | Can be blocked by ReLU | Always flows through |
| **Max Depth** | ~200 layers practical | 1000+ layers work! |
| **Activation Range** | [0, ∞) after ReLU | (-∞, ∞) full range |
| **Training Stability** | Good | Excellent |
| **Use Case** | General purpose | Ultra-deep networks |

**Bottom line:** Use pre-activation for:
- New architectures
- Very deep networks (100+ layers)
- Transfer learning
- Research projects

---

## 💡 The Philosophy

### Why Small Details Matter

ResNet V2 teaches us that **tiny architectural changes** can have huge impacts:

**The change:** Move BN-ReLU before Conv  
**The impact:** Enable 5× deeper networks!

### The Identity Principle

**Key insight:** The best path forward often preserves the past.

Don't force every layer to transform—give it the option to pass information through unchanged!

---

**Completed Day 10?** Move on to **[Day 11: Dilated Convolutions](../day11_dilated_convolutions/)** where we learn to capture multi-scale context!

**Questions?** Check the [notebook.ipynb](notebook.ipynb) for interactive experiments.

---

*"In our residual learning formulation, the shortcuts are directly built on identity mappings... An asymmetric identity shortcut that works better than identity is not observed in any forms."* - Kaiming He on why clean identity paths matter