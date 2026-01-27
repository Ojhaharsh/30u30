"""
Solution 4: Transfer Learning with AlexNet
==========================================

Fine-tune pretrained AlexNet on new task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def get_pretrained_alexnet(num_classes, freeze_features=True):
    """Load pretrained AlexNet and modify classifier."""
    model = models.alexnet(pretrained=True)
    
    # Freeze conv layers
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Replace last FC layer
    model.classifier[6] = nn.Linear(4096, num_classes)
    
    return model


def get_data_loaders(batch_size=32):
    """Get CIFAR-10 loaders with ImageNet preprocessing."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.CIFAR10('./data', True, transform, download=True)
    test_data = datasets.CIFAR10('./data', False, transform, download=True)
    
    return (DataLoader(train_data, batch_size, shuffle=True, num_workers=2),
            DataLoader(test_data, batch_size, shuffle=False, num_workers=2))


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    correct, total = 0, 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    
    return correct / total


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            _, pred = model(X).max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    
    return correct / total


def compare_transfer_vs_scratch():
    """Compare transfer learning vs training from scratch."""
    print("Transfer Learning vs From Scratch")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(64)
    
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    
    # Transfer learning
    print("\n1. Transfer Learning (frozen features):")
    model_tl = get_pretrained_alexnet(10, freeze_features=True).to(device)
    trainable = sum(p.numel() for p in model_tl.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_tl.parameters()), lr=0.001)
    
    for e in range(epochs):
        train_acc = train_one_epoch(model_tl, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model_tl, test_loader, device)
        print(f"   Epoch {e+1}: Train={train_acc:.3f}, Val={val_acc:.3f}")
    
    # From scratch
    print("\n2. From Scratch (random init):")
    model_scratch = models.alexnet(pretrained=False)
    model_scratch.classifier[6] = nn.Linear(4096, 10)
    model_scratch = model_scratch.to(device)
    
    optimizer = optim.Adam(model_scratch.parameters(), lr=0.001)
    
    for e in range(epochs):
        train_acc = train_one_epoch(model_scratch, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model_scratch, test_loader, device)
        print(f"   Epoch {e+1}: Train={train_acc:.3f}, Val={val_acc:.3f}")
    
    print("\n" + "=" * 60)
    print("Transfer learning converges much faster and achieves better accuracy!")


if __name__ == "__main__":
    compare_transfer_vs_scratch()
