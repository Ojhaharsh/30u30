"""
Exercise 2: Ablation Study
==========================

Goal: Remove AlexNet's innovations one at a time and measure impact.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AlexNetVariant(nn.Module):
    """AlexNet with configurable innovations."""
    
    def __init__(self, num_classes=10, use_relu=True, use_dropout=True):
        super().__init__()
        self.use_relu = use_relu
        self.use_dropout = use_dropout
        
        # Activation function
        if use_relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Sigmoid()  # Pre-AlexNet approach
        
        # Smaller version for CIFAR-10 (32x32 images)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            self.activation,
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            self.activation,
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            self.activation,
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        layers = [nn.Linear(256 * 4 * 4, 1024)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(self.activation)
        layers.append(nn.Linear(1024, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_cifar10_loaders(batch_size=64, use_augmentation=True):
    """Get CIFAR-10 data loaders."""
    
    # TODO 1: Define transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            # TODO: Add augmentations
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # TODO 2: Load datasets
    train_dataset = None  # TODO: datasets.CIFAR10(root='./data', train=True, ...)
    test_dataset = None   # TODO: datasets.CIFAR10(root='./data', train=False, ...)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # TODO 3: Standard training step
        optimizer.zero_grad()
        outputs = None  # TODO: model(X)
        loss = None     # TODO: criterion(outputs, y)
        # TODO: loss.backward()
        # TODO: optimizer.step()
        
        total_loss += loss.item() if loss else 0
        _, predicted = outputs.max(1) if outputs is not None else (None, None)
        if predicted is not None:
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), correct / total


def run_ablation():
    """Run ablation study."""
    print("AlexNet Ablation Study")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configurations to test
    configs = {
        'Baseline (ReLU + Dropout)': {'use_relu': True, 'use_dropout': True},
        'No ReLU (Sigmoid)': {'use_relu': False, 'use_dropout': True},
        'No Dropout': {'use_relu': True, 'use_dropout': False},
        'Minimal (No ReLU, No Dropout)': {'use_relu': False, 'use_dropout': False},
    }
    
    results = {}
    epochs = 10
    
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"Config: {config}")
        
        # TODO 4: Create model with configuration
        model = None  # TODO: AlexNetVariant(num_classes=10, **config).to(device)
        
        # TODO 5: Get data loaders
        train_loader, test_loader = None, None  # TODO
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        results[name] = {
            'train_accs': train_accs,
            'val_accs': val_accs,
            'final_train': train_accs[-1],
            'final_val': val_accs[-1]
        }
    
    # Plot results
    plot_ablation_results(results)
    print_summary(results)


def plot_ablation_results(results):
    """Plot ablation study results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, data in results.items():
        epochs = range(1, len(data['train_accs']) + 1)
        ax1.plot(epochs, data['train_accs'], label=name)
        ax2.plot(epochs, data['val_accs'], label=name)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy by Configuration')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy by Configuration')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('AlexNet Ablation Study', fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<35} {'Train Acc':<12} {'Val Acc':<12}")
    print("-" * 60)
    
    for name, data in results.items():
        print(f"{name:<35} {data['final_train']:.3f}       {data['final_val']:.3f}")
    
    print("=" * 60)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs")
    print("2. Dataset will download automatically")
    print("3. Run ablation study and analyze results")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # run_ablation()
