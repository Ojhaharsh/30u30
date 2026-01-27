"""
Solution 5: Modern Improvements to AlexNet
==========================================

Add BatchNorm, Kaiming init, AdamW, cosine LR scheduler.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math


class ClassicAlexNet(nn.Module):
    """Original AlexNet style (for CIFAR-10)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self._init_classic()
        
    def _init_classic(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class ModernAlexNet(nn.Module):
    """Modern AlexNet with BatchNorm and Kaiming init."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Less dropout needed with BN
            nn.Linear(256 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        self._init_modern()
        
    def _init_modern(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def get_loaders(batch_size=64, augment=False):
    """Get CIFAR-10 loaders."""
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
    
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    
    train = datasets.CIFAR10('./data', True, train_tf, download=True)
    test = datasets.CIFAR10('./data', False, test_tf, download=True)
    
    return DataLoader(train, batch_size, True, num_workers=2), DataLoader(test, batch_size, False, num_workers=2)


def train(model, train_loader, test_loader, epochs, optimizer, scheduler=None, device='cpu'):
    """Train and return history."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = {'train': [], 'val': [], 'lr': []}
    
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
            if scheduler and hasattr(scheduler, 'step_batch'):
                scheduler.step()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        history['train'].append(correct / total)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if scheduler and not hasattr(scheduler, 'step_batch'):
            scheduler.step()
        
        # Eval
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


def compare_classic_vs_modern():
    """Compare classic and modern training."""
    print("Classic vs Modern AlexNet")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    
    # Classic training
    print("\n1. Classic (SGD + Step LR):")
    train_loader, test_loader = get_loaders(augment=False)
    model = ClassicAlexNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    classic_hist = train(model, train_loader, test_loader, epochs, optimizer, scheduler, device)
    
    # Modern training
    print("\n2. Modern (AdamW + Cosine LR + Augmentation):")
    train_loader, test_loader = get_loaders(augment=True)
    model = ModernAlexNet()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    modern_hist = train(model, train_loader, test_loader, epochs, optimizer, scheduler, device)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(classic_hist['train'], label='Classic')
    axes[0].plot(modern_hist['train'], label='Modern')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    
    axes[1].plot(classic_hist['val'], label='Classic')
    axes[1].plot(modern_hist['val'], label='Modern')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    
    axes[2].plot(classic_hist['lr'], label='Step LR')
    axes[2].plot(modern_hist['lr'], label='Cosine LR')
    axes[2].set_title('Learning Rate')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print(f"Classic best: {max(classic_hist['val']):.3f}")
    print(f"Modern best:  {max(modern_hist['val']):.3f}")


if __name__ == "__main__":
    compare_classic_vs_modern()
