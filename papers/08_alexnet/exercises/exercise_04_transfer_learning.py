"""
Exercise 4: Transfer Learning
=============================

Goal: Fine-tune pretrained AlexNet on a new dataset.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


def get_pretrained_alexnet(num_classes, freeze_features=True):
    """
    Get pretrained AlexNet with modified classifier.
    
    Args:
        num_classes: Number of classes for new task
        freeze_features: Whether to freeze conv layers
    """
    # TODO 1: Load pretrained AlexNet
    model = None  # TODO: models.alexnet(pretrained=True)
    
    # TODO 2: Freeze feature extraction layers
    if freeze_features:
        for param in model.features.parameters():
            # TODO: param.requires_grad = False
            pass
    
    # TODO 3: Replace classifier for new task
    # Original: Linear(9216, 4096) -> Linear(4096, 4096) -> Linear(4096, 1000)
    # New: Replace last layer with Linear(4096, num_classes)
    
    # model.classifier[6] = nn.Linear(4096, num_classes)
    
    return model


def get_random_alexnet(num_classes):
    """Get randomly initialized AlexNet (no pretraining)."""
    # TODO 4: Create AlexNet without pretrained weights
    model = None  # TODO: models.alexnet(pretrained=False)
    
    # Modify classifier for num_classes
    # model.classifier[6] = nn.Linear(4096, num_classes)
    
    return model


def get_data_loaders(data_dir='./data', batch_size=32):
    """
    Get data loaders for transfer learning.
    
    Using CIFAR-10 as proxy (could use Flowers, Pets, Food, etc.)
    """
    # TODO 5: Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # TODO 6: Load dataset
    train_dataset = None  # TODO
    val_dataset = None    # TODO
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train model with transfer learning.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # TODO 7: Create optimizer
    # Only optimize parameters that require gradients
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = None  # TODO
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # TODO 8: Training step
            optimizer.zero_grad()
            outputs = None  # TODO: model(X)
            loss = None     # TODO: criterion(outputs, y)
            # TODO: loss.backward()
            # TODO: optimizer.step()
            
            train_loss += loss.item() if loss else 0
            if outputs is not None:
                _, pred = outputs.max(1)
                train_total += y.size(0)
                train_correct += pred.eq(y).sum().item()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, pred = outputs.max(1)
                val_total += y.size(0)
                val_correct += pred.eq(y).sum().item()
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        print(f"Epoch {epoch+1}: Train Acc={history['train_acc'][-1]:.3f}, Val Acc={history['val_acc'][-1]:.3f}")
    
    return history


def compare_pretrained_vs_scratch():
    """
    Compare pretrained vs randomly initialized AlexNet.
    """
    print("Transfer Learning Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_classes = 10
    epochs = 10
    
    # Get data
    train_loader, val_loader = get_data_loaders()
    
    if train_loader is None:
        print("Could not load data!")
        return
    
    # Train pretrained model
    print("\n1. Training PRETRAINED AlexNet...")
    pretrained_model = get_pretrained_alexnet(num_classes, freeze_features=True)
    pretrained_history = train_model(pretrained_model, train_loader, val_loader, 
                                      epochs=epochs, device=device)
    
    # Train from scratch
    print("\n2. Training FROM SCRATCH...")
    scratch_model = get_random_alexnet(num_classes)
    scratch_history = train_model(scratch_model, train_loader, val_loader,
                                   epochs=epochs, device=device)
    
    # Plot comparison
    plot_comparison(pretrained_history, scratch_history)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"Pretrained final val acc: {pretrained_history['val_acc'][-1]:.3f}")
    print(f"From scratch final val acc: {scratch_history['val_acc'][-1]:.3f}")
    print(f"Improvement: {pretrained_history['val_acc'][-1] - scratch_history['val_acc'][-1]:.3f}")


def plot_comparison(pretrained_history, scratch_history):
    """Plot comparison of training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(pretrained_history['train_acc']) + 1)
    
    # Accuracy
    ax1.plot(epochs, pretrained_history['val_acc'], 'b-', label='Pretrained')
    ax1.plot(epochs, scratch_history['val_acc'], 'r-', label='From Scratch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Transfer Learning vs Training from Scratch')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Loss
    ax2.plot(epochs, pretrained_history['val_loss'], 'b-', label='Pretrained')
    ax2.plot(epochs, scratch_history['val_loss'], 'r-', label='From Scratch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs")
    print("2. Run compare_pretrained_vs_scratch()")
    print("3. Pretrained should be MUCH better")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # compare_pretrained_vs_scratch()
