"""
Exercise 4: Neural Network Training Complexity
===============================================

Goal: Monitor the complexity of neural network activations during training.
Does training follow a complexity curve?

Your Task:
- Create a simple neural network
- Capture activations during training
- Compute complexity of activation patterns
- Correlate with training/validation performance

Learning Objectives:
1. How neural networks evolve internally during training
2. Internal representations at different stages
3. Connection between complexity and generalization
4. When does "emergence" happen in neural nets?

Time: 3-4 hours
Difficulty: Very Hard ⏱️⏱️⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

# Import complexity measures
try:
    from exercise_02_complexity_measure import compute_shannon_entropy
except ImportError:
    def compute_shannon_entropy(x, num_bins=20):
        """Fallback entropy calculation."""
        flat = x.flatten()
        hist, _ = np.histogram(flat, bins=num_bins)
        p = hist / np.sum(hist)
        return -np.sum(p[p > 0] * np.log(p[p > 0]))


class SimpleNet(nn.Module):
    """
    Simple neural network for classification.
    
    Architecture: Input → FC1 → ReLU → FC2 → ReLU → FC3 → Output
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # TODO 1: Define layers
        self.fc1 = None  # TODO: nn.Linear(input_size, hidden_size)
        self.fc2 = None  # TODO: nn.Linear(hidden_size, hidden_size)
        self.fc3 = None  # TODO: nn.Linear(hidden_size, output_size)
        
        # Storage for activations (populated by hooks)
        self.activations = {}
        
    def forward(self, x):
        # Flatten if needed
        x = x.view(x.size(0), -1)
        
        # TODO 2: Forward pass with activation capture
        x1 = None  # TODO: F.relu(self.fc1(x))
        self.activations['layer1'] = x1.detach().cpu().numpy()
        
        x2 = None  # TODO: F.relu(self.fc2(x1))
        self.activations['layer2'] = x2.detach().cpu().numpy()
        
        x3 = None  # TODO: self.fc3(x2)
        self.activations['layer3'] = x3.detach().cpu().numpy()
        
        return x3


def compute_activation_complexity(activations, num_bins=20):
    """
    Compute complexity of neural network activations.
    
    Args:
        activations: Dict of layer_name -> activation array (batch x features)
        
    Returns:
        Dict with complexity measures per layer
    """
    results = {}
    
    for layer_name, acts in activations.items():
        # TODO 3: Compute entropy of activations
        # Flatten across batch dimension
        entropy = None  # TODO: compute_shannon_entropy(acts, num_bins)
        
        # TODO 4: Compute "sparsity" - fraction of near-zero activations
        # After ReLU, many activations are exactly 0
        sparsity = None  # TODO: np.mean(np.abs(acts) < 0.01)
        
        # TODO 5: Compute activation variance (measure of representation diversity)
        variance = None  # TODO: np.var(acts)
        
        # TODO 6: Combine into complexity measure
        # High entropy + high variance + medium sparsity = complex representations
        complexity = None  # TODO: entropy * np.sqrt(variance) * (1 - abs(sparsity - 0.5))
        
        results[layer_name] = {
            'entropy': entropy,
            'sparsity': sparsity,
            'variance': variance,
            'complexity': complexity
        }
    
    return results


def create_synthetic_dataset(n_samples=1000, input_size=20, n_classes=5):
    """
    Create a synthetic classification dataset.
    
    Args:
        n_samples: Number of samples
        input_size: Dimension of input
        n_classes: Number of classes
        
    Returns:
        X, y tensors
    """
    # TODO 7: Create random data with class structure
    # Each class should have a "center" and samples are around it
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        # Create class center
        center = np.random.randn(input_size) * 2
        
        # Create samples around center
        n_per_class = n_samples // n_classes
        samples = center + np.random.randn(n_per_class, input_size) * 0.5
        
        X.append(samples)
        y.extend([class_idx] * n_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    return torch.FloatTensor(X), torch.LongTensor(y)


class ComplexityTracker:
    """
    Track complexity during training.
    """
    
    def __init__(self, measure_batch_size=100):
        self.measure_batch_size = measure_batch_size
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_acc': [],
            'layer1_complexity': [],
            'layer2_complexity': [],
            'layer3_complexity': [],
            'layer1_sparsity': [],
            'layer2_sparsity': [],
            'layer3_sparsity': []
        }
        
    def record(self, model, epoch, train_loss, val_acc, measure_data):
        """Record complexity at current epoch."""
        # Run forward pass on measurement data
        model.eval()
        with torch.no_grad():
            _ = model(measure_data[:self.measure_batch_size])
        
        # Compute complexity
        complexity_results = compute_activation_complexity(model.activations)
        
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_acc'].append(val_acc)
        
        for layer_name, metrics in complexity_results.items():
            self.history[f'{layer_name}_complexity'].append(metrics['complexity'])
            self.history[f'{layer_name}_sparsity'].append(metrics['sparsity'])
        
        model.train()


def train_with_complexity_tracking(model, train_loader, val_loader, 
                                   measure_data, epochs=50, lr=0.01):
    """
    Train model while tracking activation complexity.
    
    Args:
        model: Neural network
        train_loader, val_loader: Data loaders
        measure_data: Fixed batch for consistent complexity measurement
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        tracker: ComplexityTracker with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    tracker = ComplexityTracker()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # TODO 8: Standard training loop
            optimizer.zero_grad()
            outputs = None  # TODO: model(X_batch)
            loss = None  # TODO: criterion(outputs, y_batch)
            # TODO: loss.backward()
            # TODO: optimizer.step()
            
            total_loss += loss.item() if loss else 0
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        
        # Record complexity
        tracker.record(model, epoch, avg_loss, val_acc, measure_data)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.3f}")
    
    return tracker


def plot_training_complexity(tracker):
    """
    Plot training metrics and complexity evolution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = tracker.history['epoch']
    
    # Panel 1: Loss and Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, tracker.history['train_loss'], 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, tracker.history['val_acc'], 'g-', label='Val Accuracy')
    ax1_twin.set_ylabel('Accuracy', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Layer Complexities
    ax2 = axes[0, 1]
    ax2.plot(epochs, tracker.history['layer1_complexity'], label='Layer 1')
    ax2.plot(epochs, tracker.history['layer2_complexity'], label='Layer 2')
    ax2.plot(epochs, tracker.history['layer3_complexity'], label='Layer 3')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Complexity')
    ax2.set_title('Activation Complexity by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Sparsity Evolution
    ax3 = axes[1, 0]
    ax3.plot(epochs, tracker.history['layer1_sparsity'], label='Layer 1')
    ax3.plot(epochs, tracker.history['layer2_sparsity'], label='Layer 2')
    ax3.plot(epochs, tracker.history['layer3_sparsity'], label='Layer 3')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sparsity (fraction near-zero)')
    ax3.set_title('Activation Sparsity by Layer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Complexity vs Accuracy
    ax4 = axes[1, 1]
    total_complexity = np.array(tracker.history['layer1_complexity']) + \
                       np.array(tracker.history['layer2_complexity']) + \
                       np.array(tracker.history['layer3_complexity'])
    ax4.scatter(total_complexity, tracker.history['val_acc'], 
               c=epochs, cmap='viridis', alpha=0.7)
    ax4.set_xlabel('Total Complexity')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Complexity vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Neural Network Training: Complexity Evolution', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_experiment():
    """Run the full training complexity experiment."""
    if not TORCH_AVAILABLE:
        print("PyTorch required for this exercise!")
        return
    
    print("Running Neural Network Complexity Experiment...")
    print("=" * 60)
    
    # Create dataset
    X, y = create_synthetic_dataset(n_samples=2000, input_size=50, n_classes=10)
    
    # Split
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    model = SimpleNet(input_size=50, hidden_size=64, output_size=10)
    
    # Fixed data for complexity measurement
    measure_data = X_train[:100]
    
    # Train with tracking
    tracker = train_with_complexity_tracking(
        model, train_loader, val_loader, measure_data,
        epochs=100, lr=0.001
    )
    
    # Plot results
    plot_training_complexity(tracker)
    
    print("\n" + "=" * 60)
    print("Analysis Questions:")
    print("1. Does complexity peak during training?")
    print("2. Which layer develops most complexity?")
    print("3. Is complexity correlated with generalization?")
    print("4. How does sparsity relate to complexity?")
    print("=" * 60)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Complete Exercise 2 first (complexity measures)")
    print("2. Install PyTorch if not available: pip install torch")
    print("3. Fill in all TODOs")
    print("4. Run run_experiment() to see complexity evolution")
    print("=" * 60 + "\n")
    
    if TORCH_AVAILABLE:
        # Uncomment when ready:
        # run_experiment()
        pass
    else:
        print("Install PyTorch first!")
