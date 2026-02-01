# Data Directory - Day 12: Dropout

This directory stores datasets and model checkpoints for dropout experiments.

## Contents

### Datasets
Place your datasets here:

- `mnist/` - MNIST handwritten digits
- `cifar10/` - CIFAR-10 image classification
- `custom/` - Your custom datasets

### Model Checkpoints
- `checkpoints/` - Saved model weights
  - `dropout_model_p0.5.pkl`
  - `no_dropout_model.pkl`
  - `best_model.pkl`

### Experiment Logs
- `logs/` - Training logs
- `tensorboard/` - TensorBoard logs
- `results/` - Experiment results

## Automatic Dataset Download

### MNIST
```python
from train_minimal import load_mnist

# Downloads automatically if not present
X_train, y_train, X_test, y_test = load_mnist(data_dir='./data')
```

### CIFAR-10 (with PyTorch)
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    download=True,
    transform=transform
)
```

### Manual MNIST Download
```bash
# If automatic download fails
cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

## Dataset Sizes

- **MNIST**: ~60 MB (60k train + 10k test images)
- **CIFAR-10**: ~170 MB (50k train + 10k test images)
- **Fashion-MNIST**: ~60 MB (same structure as MNIST)

## Expected Directory Structure

```
data/
├── README.md           ← You are here
├── mnist/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── cifar10/
│   └── cifar-10-batches-py/
├── checkpoints/
│   ├── dropout_0.3.pkl
│   ├── dropout_0.5.pkl
│   └── dropout_0.7.pkl
├── logs/
│   └── training_log.csv
└── results/
    ├── dropout_comparison.png
    └── mc_dropout_uncertainty.png
```

## Notes

- MNIST is automatically cached as `mnist.npz` after first download
- Dropout experiments work best with moderately-sized datasets
- For larger experiments, consider using PyTorch DataLoaders
- MC Dropout uncertainty estimation works with any saved model
