# Day 8: AlexNet - Exercises

Build your understanding of revolutionary deep learning architecture that won ImageNet 2012!

---

## Exercise 1: GPU Impact Simulator ⭐⭐⭐

Understand why AlexNet's GPU acceleration was revolutionary.

**Problem**: AlexNet couldn't fit in a single GPU's 3GB memory, so it was split across 2 GPUs. Analyze how GPU parallelization improves training speed.

**Starting Code**:
```python
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class AlexNetSimplified(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def benchmark_gpu_vs_cpu():
    """Benchmark training on GPU vs CPU"""
    
    model = AlexNetSimplified()
    criterion = nn.CrossEntropyLoss()
    
    batch_sizes = [32, 64, 128, 256]
    gpu_times = []
    cpu_times = []
    
    # TODO: Time forward/backward passes on GPU and CPU
    # For each batch size, run 10 training iterations
    # Measure elapsed time using time.time()
    
    return batch_sizes, gpu_times, cpu_times

results = benchmark_gpu_vs_cpu()
# Plot comparison
```

**Expected Behavior**:
- GPU should be 10-50x faster than CPU for large batches
- Speedup increases with batch size (better GPU utilization)
- At batch size 256, GPU training should be ~30-50x faster

**Hints**:
1. Move model and data to device using `.to(device)`
2. Warm up GPU before timing to get accurate measurements
3. Use `torch.cuda.synchronize()` to ensure GPU operations complete
4. Measure total time for 10 iterations (forward + backward)
5. Compare GPU vs CPU throughput (images/second)

**Grading Rubric**:
- ⭐⭐⭐ (7/10): Correctly times GPU vs CPU, shows speedup
- ⭐⭐⭐⭐ (9/10): Also includes memory usage analysis
- ⭐⭐⭐⭐⭐ (10/10): Analyzes scaling efficiency and practical implications

---

## Exercise 2: ReLU vs Sigmoid ⭐⭐⭐⭐

AlexNet's key innovation: ReLU activation instead of sigmoid/tanh.

**Problem**: Compare how different activations affect training dynamics. Measure convergence speed, vanishing gradients, and representational power.

**Starting Code**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class NetworkWithActivation(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.act1 = activation_fn
        self.fc2 = nn.Linear(512, 256)
        self.act2 = activation_fn
        self.fc3 = nn.Linear(256, 128)
        self.act3 = activation_fn
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

def compare_activations():
    """Compare ReLU, Sigmoid, and Tanh activations"""
    
    activations = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh()
    }
    
    # Load MNIST
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
    results = {}
    
    for act_name, act_fn in activations.items():
        model = NetworkWithActivation(act_fn)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        grad_magnitudes = []
        
        # TODO: Train for 5 epochs
        # - Track training loss per batch
        # - Track average gradient magnitude in first layer
        # - Measure training time
        
        results[act_name] = {
            'losses': losses,
            'gradients': grad_magnitudes,
            'final_loss': losses[-1] if losses else None
        }
    
    return results

results = compare_activations()
# Visualize results
```

**Expected Findings**:
- ReLU: Fast convergence, large gradients, stable training
- Sigmoid/Tanh: Slow convergence, vanishing gradients, training plateaus
- ReLU is 3-5x faster to convergence on MNIST

**Hints**:
1. Use `model.fc1.weight.grad` to access gradients
2. Calculate gradient magnitude as `grad.norm().item()`
3. Track gradients at beginning of training when they're most informative
4. Plot loss curves with log scale to see convergence patterns

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Correctly compares activation functions with loss curves
- ⭐⭐⭐⭐ (9/10): Includes gradient magnitude analysis showing vanishing gradients
- ⭐⭐⭐⭐⭐ (10/10): Also analyzes dead ReLU problem and proposes solutions

---

## Exercise 3: Learning Rate Scheduling ⭐⭐⭐⭐

AlexNet trained with manual learning rate annealing - crucial for convergence.

**Problem**: Implement and test different learning rate schedules. Find which is fastest to convergence.

**Starting Code**:
```python
import torch.optim.lr_scheduler as lr_scheduler

class LRScheduleComparison:
    def __init__(self):
        self.schedules = {
            'constant': lambda opt: None,
            'step': lambda opt: lr_scheduler.StepLR(opt, step_size=10, gamma=0.1),
            'exponential': lambda opt: lr_scheduler.ExponentialLR(opt, gamma=0.95),
            'cosine': lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=50),
            'warmup_cosine': lambda opt: self._warmup_cosine(opt)
        }
    
    def _warmup_cosine(self, optimizer):
        # TODO: Implement warmup followed by cosine annealing
        # Warmup: linearly increase LR from 0 to base_lr over 5 epochs
        # Then: cosine annealing for remaining epochs
        pass
    
    def compare_schedules(self):
        # TODO: Train AlexNet with each schedule
        # Track: loss, validation accuracy, convergence speed
        # Measure: total time to reach 95% accuracy
        
        results = {}
        
        return results

comparison = LRScheduleComparison()
results = comparison.compare_schedules()
```

**Expected Results**:
- Constant LR: Training plateaus around epoch 20
- Step decay: Improves, converges by epoch 40
- Exponential: Smooth improvement, converges by epoch 35
- Cosine annealing: Best convergence, reaches target by epoch 30
- Warmup + Cosine: Fastest initial progress + stable training

**Hints**:
1. Call `scheduler.step()` after each epoch
2. Warmup phase: manually adjust optimizer's learning rate for first N epochs
3. Use `optimizer.param_groups[0]['lr']` to set custom LR
4. Plot LR schedule vs epoch to visualize all curves

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Implements 3+ schedules with proper comparison
- ⭐⭐⭐⭐ (9/10): Includes warmup scheduling and analysis
- ⭐⭐⭐⭐⭐ (10/10): Discovers that warmup+cosine is optimal for this task

---

## Exercise 4: Data Augmentation Importance ⭐⭐⭐⭐

AlexNet used aggressive data augmentation to regularize training.

**Problem**: Train models with different augmentation levels. Measure impact on overfitting and generalization.

**Starting Code**:
```python
from torchvision import transforms

# Define augmentation strategies
augmentations = {
    'none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    
    'basic': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    
    'strong': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # TODO: Add more augmentations similar to AlexNet
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    
    'alexnet_style': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # TODO: Add AlexNet-style augmentations:
        # - Random crops to 224x224
        # - PCA color augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
}

def measure_augmentation_impact():
    # TODO: Train with each augmentation level
    # Track: training loss, validation loss, train/test gap (overfitting)
    # Measure: final test accuracy for each strategy
    
    results = {}
    
    return results

results = measure_augmentation_impact()
```

**Expected Results**:
- No augmentation: Best training loss, worst test accuracy (high gap = overfitting)
- Basic: Reduced overfitting, better generalization
- Strong: Similar to basic, performance plateaus
- AlexNet-style: Best test accuracy, narrowest train/test gap

**Hints**:
1. Use same model architecture and hyperparameters for fair comparison
2. Plot train loss vs validation loss to visualize overfitting
3. Calculate "generalization gap" = test_loss - train_loss
4. AlexNet used multi-crop evaluation at test time for better accuracy

**Grading Rubric**:
- ⭐⭐⭐⭐ (8/10): Tests 3+ augmentation strategies with proper metrics
- ⭐⭐⭐⭐ (9/10): Implements AlexNet-style augmentations (crops, color jitter)
- ⭐⭐⭐⭐⭐ (10/10): Demonstrates that augmentation is essential for modern deep learning

---

## Exercise 5: Batch Normalization Retrospective ⭐⭐⭐⭐⭐

AlexNet predates batch norm (invented 2015), compare what we can do now.

**Problem**: Reimplement AlexNet with batch normalization. Measure training stability and convergence speed improvements.

**Starting Code**:
```python
class AlexNetWithoutBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ... rest of network
        )

class AlexNetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),  # ADD BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),  # ADD BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ... rest of network
        )

def compare_with_without_bn():
    # TODO: Train both models on CIFAR-10
    # Track: training loss, validation loss, training stability
    # Measure: time to convergence, final accuracy
    
    models = {
        'AlexNet (no BN)': AlexNetWithoutBN(),
        'AlexNet (with BN)': AlexNetWithBN()
    }
    
    results = {}
    
    return results

results = compare_with_without_bn()
# BN should show ~30-50% faster training!
```

**Expected Results**:
- Without BN: Unstable training, requires careful learning rate tuning
- With BN: Stable training, converges 2-5x faster, higher final accuracy
- BN reduces internal covariate shift, enabling faster learning rates

**Hints**:
1. Batch norm is typically applied before activation
2. Use same hyperparameters (learning rate, batch size) for both
3. Track gradient norm to see internal covariate shift
4. BN momentum parameter (default 0.1) affects running statistics

**Grading Rubric**:
- ⭐⭐⭐⭐ (9/10): Correctly implements both versions, shows BN improves convergence
- ⭐⭐⭐⭐⭐ (10/10): Also analyzes gradient flow and explains why BN helps

---

## Bonus Challenge: Multi-GPU Training 🚀

Implement data parallelism across multiple GPUs (or simulate with CPU).

```python
# TODO: Use nn.DataParallel to split batch across GPUs
# - Train AlexNet on CIFAR-10 with data parallelism
# - Measure speedup with 1 GPU vs 2 GPUs vs 4 GPUs (if available)
# - Show that speedup saturates due to communication overhead

# This mirrors the original AlexNet paper's approach!
```

**Key Insight**: The original AlexNet was split across 2 GPUs because a single GPU (3GB memory) couldn't fit the entire network. Modern GPUs with 8GB+ VRAM don't need this, but distributed training is still important for large-scale models!

---

## Summary

**Concepts Mastered**:
- ✅ GPU acceleration impact on deep learning
- ✅ ReLU advantages over sigmoid/tanh
- ✅ Learning rate scheduling for convergence
- ✅ Data augmentation for regularization
- ✅ Batch normalization improvements
- ✅ Multi-GPU parallelization

**Aha! Moments**:
1. GPU acceleration made deep learning practical
2. ReLU solved vanishing gradient problem
3. Learning rate scheduling beats constant LR by 2-3x
4. Data augmentation is regularization in action
5. Batch norm is the "battery included" for stable training
6. AlexNet would be trained completely differently today!

These practical insights transfer directly to modern architectures! 🔥
