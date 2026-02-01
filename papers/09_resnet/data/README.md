# Data Directory - Day 9: ResNet

This directory stores datasets and model checkpoints for ResNet training.

## Contents

### Datasets
Place your datasets here:

- `imagenet/` - ImageNet dataset (download from http://www.image-net.org/)
- `cifar10/` - CIFAR-10 dataset (auto-downloaded by torchvision)
- `cifar100/` - CIFAR-100 dataset (auto-downloaded by torchvision)
- `custom/` - Your custom image classification datasets

### Model Checkpoints
- `checkpoints/` - Saved model weights
- `pretrained/` - Pre-trained ResNet weights
  - `resnet18-*.pth`
  - `resnet34-*.pth`
  - `resnet50-*.pth`
  - `resnet101-*.pth`
  - `resnet152-*.pth`

### Logs & Visualizations
- `tensorboard/` - TensorBoard log files
- `wandb/` - Weights & Biases logs
- `gradient_flow/` - Gradient flow visualizations

## Downloading Datasets

### CIFAR-10/100 (Automatic)
```python
from torchvision import datasets

# CIFAR-10
trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True)

# CIFAR-100
trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True)
```

### Pre-trained Weights (Automatic)
```python
from torchvision import models

# PyTorch will download automatically
model = models.resnet50(pretrained=True)
```

## Notes

- CIFAR-10: ~170 MB
- CIFAR-100: ~170 MB
- ImageNet: ~150 GB
- ResNet-50 checkpoint: ~100 MB
- ResNet-152 checkpoint: ~240 MB
