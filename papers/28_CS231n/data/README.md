# Data Directory: Day 28 — CS231n CNNs for Visual Recognition

This directory stores the CIFAR-10 dataset used by `train_minimal.py`.

## Dataset

**CIFAR-10** (Canadian Institute for Advanced Research, 10 classes)
- 60,000 32x32 color images in 10 classes
- 50,000 training images, 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Source: https://www.cs.toronto.edu/~kriz/cifar.html

## Usage

The training script downloads CIFAR-10 automatically on first run:

```bash
python train_minimal.py --epochs 3
```

The download creates a `cifar-10-batches-py/` subdirectory here.

## Note on Training Speed

Our `implementation.py` uses pure NumPy for convolutions. This is intentionally slow — the goal is to understand the forward pass, not to train at production speed. The default training uses only 200 samples. For full CIFAR-10 training, use PyTorch or another framework with GPU support.
