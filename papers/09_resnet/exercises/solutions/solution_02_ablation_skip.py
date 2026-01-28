"""
Solution 2: Skip Connection Ablation
====================================

Compare plain network vs ResNet to show degradation problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class PlainBlock(nn.Module):
    """Block WITHOUT skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out  # No skip!


class ResidualBlock(nn.Module):
    """Block WITH skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)  # Skip connection!
        return out


class PlainNet(nn.Module):
    """Plain network without skip connections."""
    def __init__(self, num_blocks=10, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[PlainBlock(64) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResNet(nn.Module):
    """ResNet with skip connections."""
    def __init__(self, num_blocks=10, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_loaders(batch_size=64):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train = datasets.CIFAR10('./data', True, tf, download=True)
    test = datasets.CIFAR10('./data', False, tf, download=True)
    return DataLoader(train, batch_size, True), DataLoader(test, batch_size, False)


def train_model(model, train_loader, test_loader, epochs=20, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
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
        history['train'].append(correct / total)
        
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


def compare_plain_vs_resnet():
    """Compare training of plain vs residual networks."""
    print("Plain vs ResNet Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders()
    
    num_blocks = 15  # Deep enough to show degradation
    
    print(f"\n1. Plain Network ({num_blocks} blocks, {num_blocks*2+1} conv layers):")
    plain = PlainNet(num_blocks)
    plain_hist = train_model(plain, train_loader, test_loader, 15, device)
    
    print(f"\n2. ResNet ({num_blocks} blocks, {num_blocks*2+1} conv layers):")
    resnet = ResNet(num_blocks)
    resnet_hist = train_model(resnet, train_loader, test_loader, 15, device)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(plain_hist['val'], 'b-', label='Plain Network', linewidth=2)
    plt.plot(resnet_hist['val'], 'r-', label='ResNet', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Degradation Problem: {num_blocks*2+1}-layer Networks')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 60)
    print(f"Plain best: {max(plain_hist['val']):.3f}")
    print(f"ResNet best: {max(resnet_hist['val']):.3f}")
    print("\nSkip connections solve the degradation problem!")


if __name__ == "__main__":
    compare_plain_vs_resnet()
