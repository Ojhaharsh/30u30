"""
Exercise 5: Modern Improvements
===============================

Goal: Add modern techniques to AlexNet and measure improvements.

Time: 3-4 hours
Difficulty: Very Hard ⏱️⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math


class ClassicAlexNet(nn.Module):
    """Original AlexNet (adapted for CIFAR-10)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        
        # Original initialization
        self._initialize_weights_classic()
    
    def _initialize_weights_classic(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModernAlexNet(nn.Module):
    """
    AlexNet with modern improvements.
    
    Improvements:
    1. Batch Normalization
    2. Kaiming initialization
    3. (Optimizer and LR scheduler handled in training)
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # TODO 1: Add BatchNorm after each conv layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            None,  # TODO: nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            None,  # TODO: nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            None,  # TODO: nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # TODO 2: Add BatchNorm to classifier too
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Reduced dropout with BN
            nn.Linear(256 * 4 * 4, 1024),
            None,  # TODO: nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )
        
        # TODO 3: Kaiming initialization
        self._initialize_weights_modern()
    
    def _initialize_weights_modern(self):
        """Use Kaiming/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TODO 4: Kaiming initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                pass
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # TODO 5: Kaiming for linear too
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                pass
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CosineScheduler:
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * lr_scale


def get_data_loaders(batch_size=64, use_modern_aug=False):
    """Get CIFAR-10 loaders."""
    
    if use_modern_aug:
        # TODO 6: Modern augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # TODO: Add AutoAugment or RandAugment
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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
    
    train_dataset = datasets.CIFAR10('./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_classic(model, train_loader, val_loader, epochs=20, device='cpu'):
    """Train with classic settings (SGD, step LR)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Classic optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    return _train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device)


def train_modern(model, train_loader, val_loader, epochs=20, device='cpu'):
    """Train with modern settings (AdamW, cosine LR)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # TODO 7: Modern optimizer
    optimizer = None  # TODO: optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    total_steps = epochs * len(train_loader)
    scheduler = CosineScheduler(optimizer, total_steps, warmup_steps=total_steps // 10)
    
    return _train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, 
                  step_scheduler_per_batch=True)


def _train(model, train_loader, val_loader, criterion, optimizer, scheduler, 
           epochs, device, step_scheduler_per_batch=False):
    """Training loop."""
    history = {'train_acc': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if step_scheduler_per_batch:
                scheduler.step()
            
            _, pred = outputs.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        
        if not step_scheduler_per_batch:
            scheduler.step()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, pred = outputs.max(1)
                val_total += y.size(0)
                val_correct += pred.eq(y).sum().item()
        
        history['train_acc'].append(correct / total)
        history['val_acc'].append(val_correct / val_total)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}: Train={history['train_acc'][-1]:.3f}, Val={history['val_acc'][-1]:.3f}, LR={history['lr'][-1]:.6f}")
    
    return history


def compare_classic_vs_modern():
    """Compare classic and modern AlexNet."""
    print("Classic vs Modern AlexNet Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 30
    
    # Classic
    print("\n1. Training CLASSIC AlexNet...")
    train_loader, val_loader = get_data_loaders(use_modern_aug=False)
    classic_model = ClassicAlexNet()
    classic_history = train_classic(classic_model, train_loader, val_loader, epochs, device)
    
    # Modern
    print("\n2. Training MODERN AlexNet...")
    train_loader, val_loader = get_data_loaders(use_modern_aug=True)
    modern_model = ModernAlexNet()
    modern_history = train_modern(modern_model, train_loader, val_loader, epochs, device)
    
    # Plot
    plot_comparison(classic_history, modern_history)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"Classic best val acc: {max(classic_history['val_acc']):.3f}")
    print(f"Modern best val acc: {max(modern_history['val_acc']):.3f}")


def plot_comparison(classic, modern):
    """Plot comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(classic['train_acc']) + 1)
    
    # Training accuracy
    axes[0].plot(epochs, classic['train_acc'], 'b-', label='Classic')
    axes[0].plot(epochs, modern['train_acc'], 'r-', label='Modern')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Validation accuracy
    axes[1].plot(epochs, classic['val_acc'], 'b-', label='Classic')
    axes[1].plot(epochs, modern['val_acc'], 'r-', label='Modern')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, classic['lr'], 'b-', label='Classic (Step)')
    axes[2].plot(epochs, modern['lr'], 'r-', label='Modern (Cosine)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.suptitle('Classic vs Modern AlexNet', fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs")
    print("2. Run compare_classic_vs_modern()")
    print("3. Modern should train faster and reach higher accuracy")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # compare_classic_vs_modern()
