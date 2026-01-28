# ResNet Cheatsheet 📋

Quick reference for Deep Residual Learning for Image Recognition

---

## The Big Idea (30 seconds)

ResNet solved the **vanishing gradient problem** by introducing **skip connections** that let information flow directly through the network. Think of it as:
- **Traditional Deep Network** = Telephone game (message degrades)
- **ResNet** = Telephone + direct highway (message preserved)
- **Result** = Networks can be 100+ layers deep and still train effectively

**Magic formula**: `output = F(x) + x` (learn the residual!)

---

## Architecture: The Skip Connection Revolution

### Basic Residual Block
```
Input x
    ↓
[Conv → BatchNorm → ReLU → Conv → BatchNorm] = F(x)
    ↓                                           ↑
  F(x) ←----------- ADD ←-------------------- x (skip connection)
    ↓
  ReLU
    ↓
 Output: F(x) + x
```

### ResNet Variants
```
ResNet-18:  18 layers,  11M params,  69.8% ImageNet accuracy
ResNet-34:  34 layers,  21M params,  73.3% ImageNet accuracy  
ResNet-50:  50 layers,  25M params,  76.0% ImageNet accuracy
ResNet-101: 101 layers, 44M params,  77.4% ImageNet accuracy
ResNet-152: 152 layers, 60M params,  78.3% ImageNet accuracy
```

**Key insight**: Identity mappings make very deep networks trainable!

---

## Quick Start

### Training
```bash
# Train ResNet-50 on ImageNet
python train_minimal.py --arch resnet50 --data imagenet --epochs 90 --batch-size 256

# Fine-tune on custom dataset
python train_minimal.py --arch resnet18 --pretrained --data custom --lr 0.001
```

### Inference
```bash
# Classify with pretrained model
python train_minimal.py --predict --arch resnet50 --checkpoint resnet50.pth --image cat.jpg
```

### In Python
```python
from implementation import ResNet18, ResNet50

# Create model
model = ResNet50(num_classes=1000)

# Load pretrained
model = torchvision.models.resnet50(pretrained=True)

# Forward pass
output = model(image_tensor)
```

---

## ResNet Building Blocks

### Basic Block (ResNet-18/34)
```python
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Skip connection with projection if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection!
        out = F.relu(out)
        return out
```

### Bottleneck Block (ResNet-50/101/152)
```python
class BottleneckBlock(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv for spatial processing
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
```

---

## Why Skip Connections Work

### The Vanishing Gradient Problem
```python
# In deep networks without skip connections
gradient_layer_n = gradient_output
for layer in reversed(layers):
    gradient_layer_n *= layer_gradient  # Gets smaller each step
    
# Result: Early layers receive tiny gradients → can't learn
```

### ResNet Solution
```python
# With skip connections
# Gradient flows through TWO paths:
# 1. Through the residual function F(x)
# 2. Directly through the identity mapping x

def backward_pass():
    # Direct path: ∂E/∂x = ∂E/∂output (always flows!)
    # Residual path: ∂E/∂x += ∂E/∂F(x) * ∂F(x)/∂x
    return direct_gradient + residual_gradient
```

### Identity Mapping Insight
```python
# If F(x) = 0 (worst case), we still have:
output = F(x) + x = 0 + x = x  # Identity mapping!

# Network can never perform worse than identity
# Always has a "fallback" option
```

---

## Training Recipe

### Hyperparameters (ImageNet)
```python
lr = 0.1                     # Initial learning rate
momentum = 0.9               # SGD momentum
weight_decay = 1e-4          # L2 regularization
batch_size = 256             # Mini-batch size
epochs = 90                  # Training epochs
lr_schedule = [30, 60]       # Reduce LR at these epochs
```

### Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Learning Rate Schedule
```python
# Step decay (original ResNet)
def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in [30, 60]:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Cosine annealing (modern approach)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=0
)
```

---

## Implementation Tips

### Batch Normalization Placement
```python
# Correct order in ResNet
x = self.conv(x)
x = self.batch_norm(x)  # Always after conv
x = F.relu(x)           # Always after batch norm
```

### Weight Initialization
```python
# He initialization for ReLU networks
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

### Gradient Clipping (if needed)
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

---

## Visualization & Analysis

### Feature Map Visualization
```python
from visualization import ResNetVisualizer

viz = ResNetVisualizer(model)

# Visualize residual learning
viz.plot_residual_flow(image, layer='layer1.0')

# Show gradient flow
viz.plot_gradient_flow()

# Feature map evolution
viz.plot_feature_maps(image, ['layer1', 'layer2', 'layer3', 'layer4'])
```

### Skip Connection Analysis
```python
def analyze_skip_connections(model, input_tensor):
    """Analyze the contribution of skip connections."""
    
    # Hook to capture residuals
    residuals = {}
    
    def get_residual(name):
        def hook(module, input, output):
            residuals[name] = output.detach()
        return hook
    
    # Register hooks on residual blocks
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, BottleneckBlock)):
            module.register_forward_hook(get_residual(name))
    
    # Forward pass
    model(input_tensor)
    
    return residuals
```

---

## Common Issues & Solutions

### Training Problems
```python
# Problem: Network too deep, still vanishing gradients
# Solution: Use more skip connections or try ResNet-V2

# Problem: Slow convergence
# Solution: Learning rate warmup
def warmup_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

# Problem: Overfitting
# Solution: More data augmentation, dropout, or smaller model
transforms.RandAugment(num_ops=2, magnitude=9)
```

### Memory Issues
```python
# Use gradient checkpointing for very deep networks
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=4)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

---

## ResNet Variants

### Wide ResNet
```python
# Wider networks instead of deeper
# ResNet-50: 50 layers, normal width
# Wide-ResNet-28-10: 28 layers, 10x wider
class WideBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dropout_rate=0.0, stride=1):
        # ... normal conv layers ...
        self.dropout = nn.Dropout(p=dropout_rate)
```

### ResNeXt (Grouped Convolutions)
```python
# Use grouped convolutions for efficiency
self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, 
                      groups=32, bias=False)  # 32 groups
```

### DenseNet-style Skip Connections
```python
# Concatenate instead of adding
def forward(self, x):
    out = self.conv_block(x)
    out = torch.cat([x, out], dim=1)  # Concatenate features
    return out
```

---

## Performance Benchmarks

### ImageNet Results (2015)
```
Model         Layers  Params   Top-1 Err  Top-5 Err
ResNet-18     18      11.7M    30.24%     10.92%
ResNet-34     34      21.8M    26.70%     8.58%
ResNet-50     50      25.6M    24.01%     7.02%
ResNet-101    101     44.5M    22.44%     6.21%
ResNet-152    152     60.2M    21.69%     5.94%
```

### Modern Improvements
```python
# Better data augmentation
transforms.TrivialAugmentWide()

# Better regularization  
nn.Dropout2d(0.1)  # Spatial dropout

# Better optimization
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.05)
```

---

## Architecture Patterns

### Residual Block Template
```python
def residual_block(x, F_function):
    """General residual block pattern."""
    identity = x
    
    # Apply transformation function
    out = F_function(x)
    
    # Add skip connection
    out += identity
    
    # Apply activation
    out = F.relu(out)
    
    return out
```

### Projection Shortcut
```python
# When dimensions don't match
def projection_shortcut(x, target_channels, stride):
    """Project input to match output dimensions."""
    return nn.Conv2d(x.shape[1], target_channels, 1, stride, bias=False)(x)
```

---

## Debug Commands

### Model Analysis
```python
# Count parameters by layer
def count_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {params:,} parameters")

# Visualize architecture
from torchviz import make_dot
make_dot(model(dummy_input), params=dict(model.named_parameters()))
```

### Training Monitoring
```python
# Monitor gradient norms per layer
def monitor_gradients(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.norm().item()
    return grads

# Check for dead ReLUs
def check_activation_health(activations):
    dead_ratio = (activations == 0).float().mean()
    print(f"Dead neurons: {dead_ratio:.2%}")
```

### Skip Connection Analysis
```python
# Measure skip connection contribution
def skip_contribution(model, input_batch):
    contributions = {}
    
    def hook(module, input, output, name):
        # Assuming output = F(x) + x format
        residual_contribution = (output - input[0]).abs().mean()
        identity_contribution = input[0].abs().mean()
        contributions[name] = {
            'residual': residual_contribution.item(),
            'identity': identity_contribution.item()
        }
    
    # Register hooks and analyze
    # ... hook registration code ...
    
    return contributions
```

---

## Historical Context

### Pre-ResNet Era (2012-2015)
- **AlexNet**: 8 layers max
- **VGGNet**: 19 layers with difficulty
- **Problem**: Deeper networks performed worse (degradation problem)

### ResNet Revolution (2015)
- **Breakthrough**: 152 layers trained successfully  
- **Key insight**: Learn residuals, not direct mappings
- **Impact**: Won ImageNet 2015 with 3.57% error rate

### Post-ResNet Developments
```
2016: ResNet-V2 (pre-activation)
2016: Wide ResNet (wider > deeper)
2017: ResNeXt (grouped convolutions)
2017: DenseNet (dense connections)
2019: EfficientNet (compound scaling)
```

*ResNet didn't just solve a technical problem - it unlocked the potential of truly deep learning! 🚀*