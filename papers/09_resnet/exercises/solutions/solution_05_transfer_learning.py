"""
Solution 5: Transfer Learning with ResNet
=========================================

Fine-tune pretrained ResNet on new dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def get_pretrained_resnet(num_classes=10, freeze_backbone=True):
    """
    Load pretrained ResNet-18 and modify for new task.
    
    Args:
        num_classes: Number of output classes
        freeze_backbone: If True, only train the classifier
    """
    # Load pretrained model
    model = models.resnet18(pretrained=True)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def get_data_loaders(batch_size=32):
    """Get CIFAR-10 data loaders with ImageNet normalization."""
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, pred = outputs.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    
    return correct / total


def compare_frozen_vs_finetuned():
    """Compare frozen backbone vs full fine-tuning."""
    print("Transfer Learning Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Strategy 1: Frozen backbone (only train FC)
    print("\n1. Frozen backbone (feature extraction):")
    model_frozen = get_pretrained_resnet(10, freeze_backbone=True).to(device)
    trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_frozen.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    optimizer = optim.Adam(model_frozen.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(3):
        loss, acc = train_epoch(model_frozen, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model_frozen, test_loader, device)
        print(f"   Epoch {epoch+1}: Loss={loss:.3f}, Train={acc:.3f}, Val={val_acc:.3f}")
    
    # Strategy 2: Full fine-tuning
    print("\n2. Full fine-tuning (all layers):")
    model_full = get_pretrained_resnet(10, freeze_backbone=False).to(device)
    trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable:,} (100%)")
    
    optimizer = optim.Adam(model_full.parameters(), lr=0.0001)  # Lower LR for fine-tuning
    
    for epoch in range(3):
        loss, acc = train_epoch(model_full, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model_full, test_loader, device)
        print(f"   Epoch {epoch+1}: Loss={loss:.3f}, Train={acc:.3f}, Val={val_acc:.3f}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Frozen backbone: Fast training, fewer params, good baseline")
    print("  - Full fine-tuning: Better accuracy, needs lower LR")


if __name__ == "__main__":
    compare_frozen_vs_finetuned()
