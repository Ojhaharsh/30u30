"""
Solution 2: AlexNet Ablation Study
==================================

Compare different AlexNet configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AlexNetVariant(nn.Module):
    """AlexNet variant for ablation study."""
    
    def __init__(self, num_classes=10, use_relu=True, use_dropout=True):
        super().__init__()
        
        # Choose activation
        activation = nn.ReLU(inplace=True) if use_relu else nn.Sigmoid()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            activation,
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            activation,
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            activation,
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier with optional dropout
        layers = [nn.Linear(256 * 4 * 4, 1024)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(activation)
        layers.append(nn.Linear(1024, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_loaders(batch_size=64, augment=True):
    """Get CIFAR-10 dataloaders."""
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10('./data', True, train_tf, download=True)
    test_data = datasets.CIFAR10('./data', False, test_tf, download=True)
    
    return (DataLoader(train_data, batch_size, shuffle=True, num_workers=2),
            DataLoader(test_data, batch_size, shuffle=False, num_workers=2))


def train_and_evaluate(model, train_loader, test_loader, epochs=10, device='cpu'):
    """Train model and return history."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        history['train_acc'].append(correct / total)
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                _, pred = model(X).max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        history['val_acc'].append(correct / total)
        
        print(f"  Epoch {epoch+1}: Train={history['train_acc'][-1]:.3f}, Val={history['val_acc'][-1]:.3f}")
    
    return history


def run_ablation():
    """Run full ablation study."""
    print("AlexNet Ablation Study")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders()
    
    configs = {
        'Baseline (ReLU+Dropout)': {'use_relu': True, 'use_dropout': True},
        'No ReLU (Sigmoid)': {'use_relu': False, 'use_dropout': True},
        'No Dropout': {'use_relu': True, 'use_dropout': False},
        'Minimal': {'use_relu': False, 'use_dropout': False},
    }
    
    results = {}
    for name, cfg in configs.items():
        print(f"\n{name}:")
        model = AlexNetVariant(10, **cfg)
        results[name] = train_and_evaluate(model, train_loader, test_loader, 10, device)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for name, hist in results.items():
        plt.plot(hist['val_acc'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('AlexNet Ablation Study')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 60)
    print("Conclusions:")
    print("  - ReLU is critical (sigmoid causes vanishing gradients)")
    print("  - Dropout helps generalization")
    print("  - Both combined give best results")


if __name__ == "__main__":
    run_ablation()
