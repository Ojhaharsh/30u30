"""
train_minimal.py - Train a Simple CNN on CIFAR-10

Trains the SimpleCNN from implementation.py on CIFAR-10 using SGD.
Downloads CIFAR-10 automatically if not already present.

Usage:
    python train_minimal.py --epochs 5 --lr 0.01
    python train_minimal.py --epochs 10 --lr 0.005 --batch-size 32

Reference: CS231n Course Notes - https://cs231n.github.io/convolutional-networks/
"""

import argparse
import os
import sys
import pickle
import gzip
import urllib.request
import tarfile
import numpy as np

# Import our CNN implementation
from implementation import SimpleCNN, softmax, cross_entropy_loss


# ---------------------------------------------------------------------------
# CIFAR-10 Data Loading
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_cifar10():
    """Download CIFAR-10 if not already present."""
    tar_path = os.path.join(CIFAR10_DIR, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(CIFAR10_DIR, "cifar-10-batches-py")

    if os.path.exists(extract_dir):
        return extract_dir

    os.makedirs(CIFAR10_DIR, exist_ok=True)
    print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
    urllib.request.urlretrieve(CIFAR10_URL, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(CIFAR10_DIR)
    os.remove(tar_path)
    print("CIFAR-10 ready.")

    return extract_dir


def load_cifar10_batch(filepath):
    """Load a single CIFAR-10 batch file."""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(batch[b'labels'])
    return images, labels


def load_cifar10(data_dir=None, max_train=5000, max_test=1000):
    """
    Load CIFAR-10 dataset.

    Uses a subset by default to keep training fast with our NumPy CNN.
    A full CIFAR-10 run with NumPy convolutions would take hours.

    Args:
        data_dir: Path to cifar-10-batches-py directory
        max_train: Maximum number of training samples
        max_test: Maximum number of test samples

    Returns:
        X_train, y_train, X_test, y_test
    """
    if data_dir is None:
        data_dir = download_cifar10()

    # Load training batches
    train_images, train_labels = [], []
    for i in range(1, 6):
        filepath = os.path.join(data_dir, f"data_batch_{i}")
        if os.path.exists(filepath):
            imgs, lbls = load_cifar10_batch(filepath)
            train_images.append(imgs)
            train_labels.append(lbls)

    X_train = np.concatenate(train_images)[:max_train]
    y_train = np.concatenate(train_labels)[:max_train]

    # Load test batch
    test_path = os.path.join(data_dir, "test_batch")
    if os.path.exists(test_path):
        X_test, y_test = load_cifar10_batch(test_path)
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]
    else:
        X_test = np.random.randn(max_test, 3, 32, 32).astype(np.float32)
        y_test = np.random.randint(0, 10, max_test)

    # Normalize: zero mean, unit variance per channel
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def numerical_gradient(model, x, y, param_name, h=1e-5):
    """
    Compute gradient numerically using finite differences.

    This is slow but correct — used for training our from-scratch CNN
    since we haven't implemented analytical backprop for all layers.
    """
    param = model.params[param_name]
    grad = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        # f(x + h)
        param[idx] = old_val + h
        scores_plus = model.forward(x)
        probs_plus = softmax(scores_plus)
        loss_plus = cross_entropy_loss(probs_plus, y)

        # f(x - h)
        param[idx] = old_val - h
        scores_minus = model.forward(x)
        probs_minus = softmax(scores_minus)
        loss_minus = cross_entropy_loss(probs_minus, y)

        # Gradient
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        # Restore
        param[idx] = old_val
        it.iternext()

    return grad


def train(model, X_train, y_train, X_test, y_test,
          epochs=5, lr=0.01, batch_size=16):
    """
    Train the CNN using SGD with numerical gradients.

    Note: Numerical gradients are very slow. This is intentional — the goal
    is to understand the forward pass, not to build a production trainer.
    For a fast CNN, use PyTorch (see exercises).

    Args:
        model: SimpleCNN instance
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Mini-batch size
    """
    print("=" * 60)
    print("Training SimpleCNN on CIFAR-10")
    print("=" * 60)
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Parameters:    {model.count_parameters():,}")
    print(f"  Epochs:        {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size:    {batch_size}")
    print()

    n_batches = max(1, len(X_train) // batch_size)

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_correct = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(X_train))
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            scores = model.forward(X_batch)
            probs = softmax(scores)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss

            # Predictions
            preds = np.argmax(scores, axis=1)
            epoch_correct += np.sum(preds == y_batch)

            # Backward pass (numerical gradients — slow but correct)
            for param_name in model.params:
                grad = numerical_gradient(model, X_batch, y_batch, param_name)
                model.params[param_name] -= lr * grad

            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{n_batches}, "
                      f"Loss: {loss:.4f}")

        # Epoch summary
        train_acc = epoch_correct / len(X_train)
        avg_loss = epoch_loss / n_batches

        # Test accuracy
        test_scores = model.forward(X_test[:100])
        test_preds = np.argmax(test_scores, axis=1)
        test_acc = np.mean(test_preds == y_test[:100])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss:    {avg_loss:.4f}")
        print(f"  Train Acc:   {train_acc:.2%}")
        print(f"  Test Acc:    {test_acc:.2%}")
        print()

    print("Training complete.")
    print("Note: Numerical gradients are very slow. For practical training,")
    print("use PyTorch or another framework with automatic differentiation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a simple CNN on CIFAR-10 (Day 28)"
    )
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--max-train', type=int, default=200,
                        help='Max training samples (default: 200, keep small for NumPy)')
    parser.add_argument('--max-test', type=int, default=50,
                        help='Max test samples (default: 50)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CIFAR-10 data directory')

    args = parser.parse_args()

    # Load data
    try:
        X_train, y_train, X_test, y_test = load_cifar10(
            data_dir=args.data,
            max_train=args.max_train,
            max_test=args.max_test
        )
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        print("Generating synthetic data for demonstration...")
        X_train = np.random.randn(args.max_train, 3, 32, 32).astype(np.float32)
        y_train = np.random.randint(0, 10, args.max_train)
        X_test = np.random.randn(args.max_test, 3, 32, 32).astype(np.float32)
        y_test = np.random.randint(0, 10, args.max_test)

    # Create and train model
    model = SimpleCNN(num_classes=10)
    train(model, X_train, y_train, X_test, y_test,
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
