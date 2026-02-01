# Data Directory - Day 8: AlexNet

This directory stores datasets and model checkpoints for AlexNet training.

## Contents

### Datasets
Place your datasets here:

- `imagenet/` - ImageNet dataset (download from http://www.image-net.org/)
- `cifar10/` - CIFAR-10 dataset (auto-downloaded by torchvision)
- `custom/` - Your custom image classification datasets

### Model Checkpoints
- `checkpoints/` - Saved model weights
- `pretrained/` - Pre-trained model weights

### Logs
- `tensorboard/` - TensorBoard log files
- `wandb/` - Weights & Biases logs

## Downloading Datasets

### CIFAR-10 (Automatic)
```python
from torchvision import datasets

trainset = datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    download=True
)
```

### ImageNet (Manual)
1. Register at http://www.image-net.org/
2. Download ILSVRC2012 dataset
3. Extract to `data/imagenet/`
4. Structure:
   ```
   imagenet/
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ...
   ```

## Notes

- CIFAR-10: ~170 MB
- ImageNet: ~150 GB
- Model checkpoints: ~200 MB per model
