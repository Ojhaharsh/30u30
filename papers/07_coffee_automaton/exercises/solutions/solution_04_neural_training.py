"""
Solution 4: Neural Network Training Complexity
==============================================

Track activation complexity during neural network training.
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


def compute_shannon_entropy(x, num_bins=20):
    """Entropy calculation."""
    flat = x.flatten()
    if np.max(flat) == np.min(flat):
        return 0.0
    hist, _ = np.histogram(flat, bins=num_bins)
    p = hist / np.sum(hist)
    return -np.sum(p[p > 0] * np.log(p[p > 0]))


class SimpleNet(nn.Module):
    """Simple classifier with activation capture."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activations = {}
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x1 = F.relu(self.fc1(x))
        self.activations['layer1'] = x1.detach().cpu().numpy()
        
        x2 = F.relu(self.fc2(x1))
        self.activations['layer2'] = x2.detach().cpu().numpy()
        
        x3 = self.fc3(x2)
        self.activations['layer3'] = x3.detach().cpu().numpy()
        
        return x3


def compute_activation_complexity(activations, num_bins=20):
    """Compute complexity of activations."""
    results = {}
    
    for name, acts in activations.items():
        entropy = compute_shannon_entropy(acts, num_bins)
        sparsity = np.mean(np.abs(acts) < 0.01)
        variance = np.var(acts)
        
        # Complexity peaks at medium sparsity
        complexity = entropy * np.sqrt(variance + 1e-10) * (1 - abs(sparsity - 0.5) * 2)
        
        results[name] = {
            'entropy': entropy,
            'sparsity': sparsity,
            'variance': variance,
            'complexity': complexity if np.isfinite(complexity) else 0
        }
    
    return results


def create_dataset(n_samples=1000, input_size=20, n_classes=5):
    """Create synthetic classification data."""
    X, y = [], []
    
    for c in range(n_classes):
        center = np.random.randn(input_size) * 2
        samples = center + np.random.randn(n_samples // n_classes, input_size) * 0.5
        X.append(samples)
        y.extend([c] * (n_samples // n_classes))
    
    X = np.vstack(X)
    y = np.array(y)
    idx = np.random.permutation(len(y))
    
    return torch.FloatTensor(X[idx]), torch.LongTensor(y[idx])


class ComplexityTracker:
    """Track complexity during training."""
    
    def __init__(self):
        self.history = {k: [] for k in [
            'epoch', 'train_loss', 'val_acc',
            'layer1_complexity', 'layer2_complexity', 'layer3_complexity',
            'layer1_sparsity', 'layer2_sparsity', 'layer3_sparsity'
        ]}
        
    def record(self, model, epoch, train_loss, val_acc, measure_data):
        model.eval()
        with torch.no_grad():
            _ = model(measure_data[:100])
        
        metrics = compute_activation_complexity(model.activations)
        
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_acc'].append(val_acc)
        
        for name, m in metrics.items():
            self.history[f'{name}_complexity'].append(m['complexity'])
            self.history[f'{name}_sparsity'].append(m['sparsity'])
        
        model.train()


def train_with_tracking(model, train_loader, val_loader, measure_data, 
                        epochs=50, lr=0.01):
    """Train with complexity tracking."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    tracker = ComplexityTracker()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, pred = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (pred == y_batch).sum().item()
        
        val_acc = correct / total
        tracker.record(model, epoch, avg_loss, val_acc, measure_data)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={val_acc:.3f}")
    
    return tracker


def plot_results(tracker):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = tracker.history['epoch']
    
    # Loss & Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, tracker.history['train_loss'], 'b-', label='Loss')
    ax1.set_ylabel('Loss', color='b')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, tracker.history['val_acc'], 'g-', label='Accuracy')
    ax1_twin.set_ylabel('Accuracy', color='g')
    ax1.set_title('Training Progress')
    ax1.grid(alpha=0.3)
    
    # Complexity by layer
    ax2 = axes[0, 1]
    for layer in ['layer1', 'layer2', 'layer3']:
        ax2.plot(epochs, tracker.history[f'{layer}_complexity'], label=layer)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Complexity')
    ax2.set_title('Activation Complexity')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Sparsity
    ax3 = axes[1, 0]
    for layer in ['layer1', 'layer2', 'layer3']:
        ax3.plot(epochs, tracker.history[f'{layer}_sparsity'], label=layer)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sparsity')
    ax3.set_title('Activation Sparsity')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Complexity vs Accuracy
    ax4 = axes[1, 1]
    total_c = np.sum([tracker.history[f'{l}_complexity'] for l in 
                      ['layer1', 'layer2', 'layer3']], axis=0)
    ax4.scatter(total_c, tracker.history['val_acc'], c=epochs, cmap='viridis')
    ax4.set_xlabel('Total Complexity')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Complexity vs Performance')
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Neural Network Training Complexity', fontweight='bold')
    plt.tight_layout()
    plt.show()


def demo():
    if not TORCH_AVAILABLE:
        print("PyTorch required!")
        return
    
    print("Neural Network Complexity Demo")
    print("=" * 60)
    
    X, y = create_dataset(2000, 50, 10)
    split = int(0.8 * len(X))
    
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), 32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[split:], y[split:]), 32)
    
    model = SimpleNet(50, 64, 10)
    tracker = train_with_tracking(model, train_loader, val_loader, X[:100], 
                                   epochs=100, lr=0.001)
    
    plot_results(tracker)


if __name__ == "__main__":
    demo()
