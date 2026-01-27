# AlexNet Cheatsheet 📋

Quick reference for ImageNet Classification with Deep Convolutional Neural Networks

---

## The Big Idea (30 seconds)

AlexNet proved that **bigger networks + more data + GPUs = breakthrough performance**. It won ImageNet 2012 by a massive margin, launching the deep learning revolution. Think of it as:
- **Before AlexNet** = Hand-crafted features + shallow learning
- **AlexNet** = Learn features directly from raw pixels with deep networks
- **Impact** = Changed computer vision forever

---

## Architecture: The 8-Layer Pioneer

```
Input: 224×224×3 RGB image

CONV1: 96 filters, 11×11, stride 4 → 55×55×96
POOL1: 3×3 max pooling, stride 2 → 27×27×96

CONV2: 256 filters, 5×5, pad 2 → 27×27×256  
POOL2: 3×3 max pooling, stride 2 → 13×13×256

CONV3: 384 filters, 3×3, pad 1 → 13×13×384
CONV4: 384 filters, 3×3, pad 1 → 13×13×384
CONV5: 256 filters, 3×3, pad 1 → 13×13×256
POOL5: 3×3 max pooling, stride 2 → 6×6×256

FC6: 4096 units (Dropout 0.5)
FC7: 4096 units (Dropout 0.5)  
FC8: 1000 units (ImageNet classes)
```

**Key insight**: Large kernels (11×11, 5×5) capture more spatial context!

---

## Quick Start

### Training
```bash
# Train on ImageNet subset
python train_minimal.py --data imagenet_subset --epochs 90 --batch-size 128

# Custom dataset
python train_minimal.py --data custom_dataset --lr 0.01 --momentum 0.9
```

### Inference
```bash
# Classify single image
python train_minimal.py --predict --image cat.jpg --checkpoint alexnet_model.pth
```

### In Python
```python
from implementation import AlexNet

# Create model
model = AlexNet(num_classes=1000)

# Load pretrained weights
model.load_state_dict(torch.load('alexnet.pth'))

# Classify image
prediction = model(image_tensor)
```

---

## Revolutionary Techniques

### 1. ReLU Activation
```python
# Instead of sigmoid/tanh
output = torch.relu(input)  # Much faster training!
```

### 2. Dropout Regularization  
```python
# Prevent overfitting in fully connected layers
x = F.dropout(x, p=0.5, training=self.training)
```

### 3. Data Augmentation
```python
# Horizontal flips + random crops + color jittering
transforms.RandomHorizontalFlip(),
transforms.RandomResizedCrop(224),
transforms.ColorJitter(0.4, 0.4, 0.4)
```

### 4. GPU Parallelism
```python
# Split model across 2 GPUs (original approach)
conv1_2 = nn.DataParallel(conv_layers[:2])
conv3_5 = nn.DataParallel(conv_layers[2:])
```

---

## Training Recipe

### Hyperparameters
```python
lr = 0.01                    # Learning rate
momentum = 0.9              # SGD momentum  
weight_decay = 0.0005       # L2 regularization
batch_size = 128            # Mini-batch size
epochs = 90                 # Training epochs
lr_schedule = [30, 60]      # Decay at epochs 30, 60
```

### Learning Rate Schedule
```python
# Reduce LR by 10x at specific epochs
if epoch in [30, 60]:
    lr *= 0.1
    optimizer = torch.optim.SGD(params, lr=lr)
```

### Data Augmentation Pipeline
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

## Implementation Tips

### Memory Optimization
```python
# Use gradient checkpointing for large images
model = torch.utils.checkpoint.checkpoint_sequential(model, 2, input)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Feature Extraction
```python
# Use pretrained AlexNet as feature extractor
model = torchvision.models.alexnet(pretrained=True)
model.classifier = model.classifier[:-1]  # Remove last layer

# Extract features
features = model(images)  # 4096-dimensional features
```

### Fine-tuning
```python
# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Only train classifier
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001)
```

---

## Visualization Tools

### Filter Visualization
```python
from visualization import AlexNetVisualizer

viz = AlexNetVisualizer(model)

# Visualize learned filters
viz.plot_conv_filters(layer='conv1', num_filters=16)

# Feature maps for input image
viz.plot_feature_maps(image, layer='conv2')
```

### Activation Patterns
```python
# Hook to capture intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.features[0].register_forward_hook(get_activation('conv1'))
```

---

## Common Issues & Solutions

### Training Problems
```python
# Problem: Exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Problem: Slow convergence  
# Solution: Proper weight initialization
nn.init.normal_(m.weight, mean=0, std=0.01)

# Problem: Overfitting
# Solution: More dropout, data augmentation
dropout_rate = 0.7  # Increase from 0.5
```

### Memory Issues
```python
# Reduce batch size for limited GPU memory
batch_size = 64  # Instead of 128

# Use gradient accumulation
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()

if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## Performance Benchmarks

### Original Results (2012)
- **Top-1 Error**: 37.5% (ImageNet validation)
- **Top-5 Error**: 17.0% (ImageNet validation)  
- **Training Time**: 5-6 days on 2 GTX 580 GPUs

### Modern Improvements
```python
# Better initialization
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Better optimization
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

# Better augmentation
transforms.RandAugment(num_ops=2, magnitude=9)
```

---

## Architecture Variants

### AlexNet-Mini (for smaller datasets)
```python
class AlexNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 11, 4, 2),  # Fewer filters
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # ... reduced architecture
        )
```

### AlexNet + Batch Normalization
```python
# Add batch norm for faster training
nn.Conv2d(3, 96, 11, 4, 2),
nn.BatchNorm2d(96),  # Add after conv
nn.ReLU(inplace=True),
```

---

## Debug Commands

### Model Summary
```python
from torchsummary import summary
summary(model, (3, 224, 224))  # Print architecture

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~61M for AlexNet
```

### Training Monitoring
```python
# Check gradient norms
total_norm = 0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm: {total_norm}")
```

### Activation Statistics
```python
# Monitor activation distributions
def print_activation_stats(module, input, output):
    print(f"{module.__class__.__name__}:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std:  {output.std().item():.4f}")
    print(f"  Dead: {(output == 0).float().mean().item():.4f}")
```

---

## Historical Context

### Why AlexNet Matters
- **First deep CNN**: Proved depth matters for image recognition
- **GPU revolution**: Showed specialized hardware enables bigger models  
- **Data hunger**: Demonstrated need for large datasets (ImageNet)
- **End-to-end learning**: Features learned, not hand-crafted

### What It Started
```
2012: AlexNet → Deep learning renaissance
2014: VGGNet → Deeper networks  
2015: ResNet → Even deeper with skip connections
2016: DenseNet → Maximum information flow
Today: Vision Transformers → Attention-based vision
```

*AlexNet didn't just win a competition - it launched the deep learning revolution! 🚀*