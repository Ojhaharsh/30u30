"""
Solution 3: Ultra-Deep Training
===============================

Train networks with 100+ layers using pre-activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class PreActBlock(nn.Module):
    """Pre-activation block for very deep networks."""
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


class UltraDeepResNet(nn.Module):
    """Very deep ResNet for CIFAR-10."""
    
    def __init__(self, num_blocks=50, channels=64, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[PreActBlock(channels) for _ in range(num_blocks)])
        self.bn_final = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)
        
        self.depth = num_blocks * 2 + 2  # conv layers
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = F.relu(self.bn_final(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    
    train = datasets.CIFAR10('./data', True, tf, download=True)
    test = datasets.CIFAR10('./data', False, test_tf, download=True)
    return DataLoader(train, batch_size, True), DataLoader(test, batch_size, False)


def train_model(model, train_loader, test_loader, epochs, device):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            _, pred = model(X).max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        history['train'].append(correct / total)
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                _, pred = model(X).max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        history['val'].append(correct / total)
        
        print(f"  Epoch {epoch+1}: Train={history['train'][-1]:.3f}, Val={history['val'][-1]:.3f}")
    
    return history


def test_ultra_deep():
    """Test training very deep networks."""
    print("Ultra-Deep Network Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders()
    
    depths = [20, 50, 100]  # Number of blocks (conv layers = 2*blocks + 2)
    results = {}
    
    for num_blocks in depths:
        model = UltraDeepResNet(num_blocks=num_blocks)
        print(f"\nTraining {model.depth}-layer network ({num_blocks} blocks):")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        history = train_model(model, train_loader, test_loader, 15, device)
        results[model.depth] = history
    
    # Plot
    plt.figure(figsize=(10, 5))
    for depth, hist in results.items():
        plt.plot(hist['val'], label=f'{depth} layers')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Ultra-Deep Networks: Pre-Activation Enables 100+ Layers')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 60)
    for depth, hist in results.items():
        print(f"{depth:3d} layers: best val = {max(hist['val']):.3f}")


if __name__ == "__main__":
    test_ultra_deep()
