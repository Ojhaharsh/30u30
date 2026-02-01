# Data Directory - Day 10: ResNet V2

This directory stores datasets and model checkpoints for ResNet V2 (Pre-activation) training.

## Contents

### Datasets
Place your datasets here:

- `imagenet/` - ImageNet dataset
- `cifar10/` - CIFAR-10 dataset (auto-downloaded)
- `cifar100/` - CIFAR-100 dataset (auto-downloaded)
- `custom/` - Your custom datasets

### Model Checkpoints
- `checkpoints/` - Saved pre-activation ResNet weights
- `pretrained/` - Pre-trained weights
  - `preact_resnet18-*.pth`
  - `preact_resnet34-*.pth`
  - `preact_resnet50-*.pth`
  - `preact_resnet101-*.pth`
  - `preact_resnet152-*.pth`
  - `preact_resnet1001-*.pth` (ultra-deep!)

### Logs & Analysis
- `tensorboard/` - TensorBoard logs
- `wandb/` - Weights & Biases logs
- `gradient_flow/` - Gradient flow comparisons
- `activation_dist/` - Activation distribution plots

## Downloading Datasets

### CIFAR (Automatic)
```python
from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2023, 0.1994, 0.2010))
])

trainset = datasets.CIFAR10(root='./data/cifar10', train=True, 
                           download=True, transform=transform)
```

## Notes

- Pre-activation ResNet enables 1000+ layer networks
- Ultra-deep models (ResNet-1001) require ~500 MB storage
- Training 1000-layer networks needs significant GPU memory
- Use gradient checkpointing for memory efficiency
