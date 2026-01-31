"""
Minimal Training Script for Dropout Experiments

Train neural networks with dropout on MNIST to see the regularization effect.
Contains everything you need to experiment with dropout rates.

Usage:
    python train_minimal.py --dropout 0.5 --epochs 20
    python train_minimal.py --compare-rates
    python train_minimal.py --mc-dropout --samples 100

Author: 30u30 Project
"""

import numpy as np
import argparse
import pickle
import os
from typing import Tuple, List, Optional
import time


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mnist(data_dir: str = 'data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset.
    
    If not available, generates synthetic data for testing.
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    mnist_path = os.path.join(data_dir, 'mnist.npz')
    
    if os.path.exists(mnist_path):
        data = np.load(mnist_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    # Try to download
    try:
        print("Downloading MNIST...")
        from urllib.request import urlretrieve
        import gzip
        
        # Download from keras source
        base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
        
        os.makedirs(data_dir, exist_ok=True)
        
        def load_images(filename):
            path = os.path.join(data_dir, filename)
            urlretrieve(base_url + filename, path)
            with gzip.open(path, 'rb') as f:
                return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        
        def load_labels(filename):
            path = os.path.join(data_dir, filename)
            urlretrieve(base_url + filename, path)
            with gzip.open(path, 'rb') as f:
                return np.frombuffer(f.read(), np.uint8, offset=8)
        
        X_train = load_images(files['train_images']).astype(np.float32) / 255.0
        y_train = load_labels(files['train_labels'])
        X_test = load_images(files['test_images']).astype(np.float32) / 255.0
        y_test = load_labels(files['test_labels'])
        
        # Cache
        np.savez(mnist_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"Could not download MNIST: {e}")
        print("Generating synthetic data...")
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(10000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 10000)
        X_test = np.random.randn(1000, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, 1000)
        
        return X_train, y_train, X_test, y_test


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class Dropout:
    """Inverted dropout layer."""
    
    def __init__(self, p: float = 0.5):
        self.p = p  # Keep probability
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x
        self.mask = (np.random.rand(*x.shape) < self.p) / self.p
        return x * self.mask
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad
        return grad * self.mask


class Linear:
    """Linear layer with He initialization."""
    
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W.T + self.b
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW = grad.T @ self.x
        self.db = grad.sum(axis=0)
        return grad @ self.W


class ReLU:
    """ReLU activation."""
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Cross-entropy loss and gradient."""
    batch_size = probs.shape[0]
    
    # Clip for numerical stability
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1 - eps)
    
    # Loss
    log_probs = np.log(probs_clipped)
    loss = -log_probs[np.arange(batch_size), labels].mean()
    
    # Gradient
    grad = probs_clipped.copy()
    grad[np.arange(batch_size), labels] -= 1
    grad /= batch_size
    
    return loss, grad


class DropoutMLP:
    """Multi-layer perceptron with dropout."""
    
    def __init__(self, 
                 input_size: int = 784,
                 hidden_sizes: List[int] = [512, 256],
                 output_size: int = 10,
                 dropout_p: float = 0.5,
                 input_dropout_p: float = 0.8):
        """
        Args:
            dropout_p: Keep probability for hidden layers
            input_dropout_p: Keep probability for input layer (typically higher)
        """
        self.layers = []
        self.dropouts = []
        
        # Input dropout (lighter)
        self.input_dropout = Dropout(p=input_dropout_p) if input_dropout_p < 1.0 else None
        
        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # No dropout after last layer
                self.layers.append(ReLU())
                self.dropouts.append(Dropout(p=dropout_p))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Input dropout
        if self.input_dropout and self.input_dropout.training:
            x = self.input_dropout.forward(x)
        
        dropout_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            
            # Apply dropout after ReLU
            if isinstance(layer, ReLU) and dropout_idx < len(self.dropouts):
                x = self.dropouts[dropout_idx].forward(x)
                dropout_idx += 1
        
        return softmax(x)
    
    def backward(self, grad: np.ndarray) -> None:
        dropout_idx = len(self.dropouts) - 1
        
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Dropout gradient before ReLU
            if isinstance(layer, ReLU) and dropout_idx >= 0:
                grad = self.dropouts[dropout_idx].backward(grad)
                dropout_idx -= 1
            
            grad = layer.backward(grad)
    
    def train(self):
        for dropout in self.dropouts:
            dropout.training = True
        if self.input_dropout:
            self.input_dropout.training = True
    
    def eval(self):
        for dropout in self.dropouts:
            dropout.training = False
        if self.input_dropout:
            self.input_dropout.training = False
    
    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append((layer.W, layer.b))
        return params
    
    def get_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all gradients."""
        grads = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                grads.append((layer.dW, layer.db))
        return grads


class SGD:
    """SGD optimizer with momentum and weight decay."""
    
    def __init__(self, params: List, lr: float = 0.01, 
                 momentum: float = 0.9, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Velocity for momentum
        self.velocity = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    
    def step(self, grads: List[Tuple[np.ndarray, np.ndarray]]):
        for i, ((W, b), (dW, db)) in enumerate(zip(self.params, grads)):
            # Weight decay
            if self.weight_decay > 0:
                dW = dW + self.weight_decay * W
            
            # Momentum update
            vW, vb = self.velocity[i]
            vW = self.momentum * vW - self.lr * dW
            vb = self.momentum * vb - self.lr * db
            self.velocity[i] = (vW, vb)
            
            # Update parameters
            W += vW
            b += vb


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model: DropoutMLP, 
                X: np.ndarray, 
                y: np.ndarray,
                optimizer: SGD,
                batch_size: int = 64) -> float:
    """Train for one epoch."""
    model.train()
    
    indices = np.random.permutation(len(X))
    total_loss = 0
    n_batches = 0
    
    for i in range(0, len(X), batch_size):
        batch_idx = indices[i:i + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        # Forward
        probs = model.forward(X_batch)
        
        # Loss
        loss, grad = cross_entropy_loss(probs, y_batch)
        total_loss += loss
        n_batches += 1
        
        # Backward
        model.backward(grad)
        
        # Update
        optimizer.step(model.get_grads())
    
    return total_loss / n_batches


def evaluate(model: DropoutMLP, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Evaluate model accuracy and loss."""
    model.eval()
    
    probs = model.forward(X)
    loss, _ = cross_entropy_loss(probs, y)
    
    predictions = probs.argmax(axis=1)
    accuracy = (predictions == y).mean()
    
    return accuracy, loss


def mc_dropout_evaluate(model: DropoutMLP, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        n_samples: int = 100) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate with MC Dropout for uncertainty estimation.
    
    Returns:
        accuracy, mean_uncertainty, uncertainties_per_sample
    """
    model.train()  # Keep dropout active!
    
    all_probs = []
    for _ in range(n_samples):
        probs = model.forward(X)
        all_probs.append(probs)
    
    # Stack and compute statistics
    all_probs = np.stack(all_probs, axis=0)  # (n_samples, batch, classes)
    
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    
    # Accuracy from mean prediction
    predictions = mean_probs.argmax(axis=1)
    accuracy = (predictions == y).mean()
    
    # Uncertainty: average std across classes
    uncertainties = std_probs.mean(axis=1)
    mean_uncertainty = uncertainties.mean()
    
    return accuracy, mean_uncertainty, uncertainties


def train(args):
    """Main training function."""
    print("=" * 60)
    print("DROPOUT TRAINING")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    
    # Use subset for faster experiments
    if args.subset:
        X_train = X_train[:args.subset]
        y_train = y_train[:args.subset]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print(f"\nCreating model with dropout p={args.dropout}...")
    model = DropoutMLP(
        input_size=784,
        hidden_sizes=[512, 256],
        output_size=10,
        dropout_p=args.dropout,
        input_dropout_p=0.8 if args.dropout < 1.0 else 1.0
    )
    
    # Optimizer
    optimizer = SGD(
        model.get_params(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, X_train, y_train, optimizer, args.batch_size)
        
        # Evaluate
        train_acc, _ = evaluate(model, X_train, y_train)
        test_acc, test_loss = evaluate(model, X_test, y_test)
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
              f"Gap: {train_acc - test_acc:.3f} | "
              f"Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")
    
    print("-" * 60)
    
    # Final results
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    gap = final_train_acc - final_test_acc
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Test Accuracy:  {final_test_acc:.4f}")
    print(f"  Gap:            {gap:.4f}")
    
    if gap > 0.1:
        print("  ⚠️  Large gap suggests overfitting. Try increasing dropout.")
    elif gap < 0.02:
        print("  ✓  Small gap suggests good generalization!")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.data_dir, 'dropout_model.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({'model': model, 'history': history}, f)
        print(f"\nModel saved to {save_path}")
    
    return model, history


def compare_dropout_rates(args):
    """Compare different dropout rates."""
    print("=" * 60)
    print("COMPARING DROPOUT RATES")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    
    # Use smaller subset for faster comparison
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    dropout_rates = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for p in dropout_rates:
        print(f"\n--- Training with dropout keep_prob = {p} ---")
        
        keep_prob = p if p > 0 else 1.0
        
        model = DropoutMLP(
            input_size=784,
            hidden_sizes=[512, 256],
            output_size=10,
            dropout_p=keep_prob,
            input_dropout_p=1.0 if p == 0 else 0.9
        )
        
        optimizer = SGD(model.get_params(), lr=0.01, momentum=0.9)
        
        for epoch in range(20):
            train_epoch(model, X_train, y_train, optimizer, 64)
        
        train_acc, train_loss = evaluate(model, X_train, y_train)
        test_acc, test_loss = evaluate(model, X_test, y_test)
        
        results[p] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc
        }
        
        print(f"  Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {train_acc - test_acc:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dropout p':<12} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<12}")
    print("-" * 48)
    
    for p, r in sorted(results.items()):
        gap_indicator = "←" if r['gap'] < 0.05 else ""
        print(f"{p:<12.1f} {r['train_acc']:<12.3f} {r['test_acc']:<12.3f} {r['gap']:<12.3f} {gap_indicator}")
    
    # Find best
    best_p = max(results.keys(), key=lambda p: results[p]['test_acc'])
    print(f"\n✓ Best test accuracy: p={best_p} with {results[best_p]['test_acc']:.3f}")
    
    return results


def run_mc_dropout(args):
    """Demonstrate MC Dropout for uncertainty."""
    print("=" * 60)
    print("MC DROPOUT - UNCERTAINTY ESTIMATION")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    
    # Train a model with dropout
    print("\nTraining model with dropout...")
    model = DropoutMLP(
        input_size=784,
        hidden_sizes=[512, 256],
        output_size=10,
        dropout_p=0.5
    )
    
    optimizer = SGD(model.get_params(), lr=0.01, momentum=0.9)
    
    for epoch in range(10):
        train_epoch(model, X_train[:5000], y_train[:5000], optimizer, 64)
    
    print(f"Training complete. Running MC Dropout with {args.mc_samples} samples...")
    
    # MC Dropout on test set
    test_subset = X_test[:100]
    labels_subset = y_test[:100]
    
    accuracy, mean_uncertainty, uncertainties = mc_dropout_evaluate(
        model, test_subset, labels_subset, n_samples=args.mc_samples
    )
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Mean Uncertainty: {mean_uncertainty:.4f}")
    
    # Find high vs low uncertainty samples
    high_uncertainty_idx = uncertainties.argsort()[-5:]
    low_uncertainty_idx = uncertainties.argsort()[:5]
    
    print(f"\n  High uncertainty samples (least confident):")
    for idx in high_uncertainty_idx:
        print(f"    Sample {idx}: uncertainty = {uncertainties[idx]:.4f}, "
              f"true label = {labels_subset[idx]}")
    
    print(f"\n  Low uncertainty samples (most confident):")
    for idx in low_uncertainty_idx:
        print(f"    Sample {idx}: uncertainty = {uncertainties[idx]:.4f}, "
              f"true label = {labels_subset[idx]}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with Dropout')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of training data')
    
    # Model
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout keep probability (0.5 = drop 50%%)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization')
    
    # Modes
    parser.add_argument('--compare-rates', action='store_true',
                        help='Compare different dropout rates')
    parser.add_argument('--mc-dropout', action='store_true',
                        help='Run MC Dropout experiment')
    parser.add_argument('--mc-samples', type=int, default=100,
                        help='Number of MC Dropout samples')
    
    # Output
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model')
    
    args = parser.parse_args()
    
    if args.compare_rates:
        compare_dropout_rates(args)
    elif args.mc_dropout:
        run_mc_dropout(args)
    else:
        train(args)
