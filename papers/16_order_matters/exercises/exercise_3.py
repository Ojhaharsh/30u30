"""
Exercise 3: Training Pointer Networks for Sorting
==================================================

Establish a training loop for teaching a Pointer Network to perform
ascending numerical sort.

Task: Map a random list (e.g., [5.2, 1.8, 9.3]) to pointer indices 
      (e.g., [1, 0, 2]).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from implementation import PointerNetwork


class SortingDataset(Dataset):
    """Dataset for numerical sorting tasks."""
    
    def __init__(self, num_samples, set_size, min_val=0.0, max_val=10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # TODO: Generate random numbers in range [min, max]
        values = None 
        
        # TODO: Get sorted indices (ground truth)
        _, target = None, None 
        
        # Format as [set_size, 1]
        inputs = None 
        
        return inputs, target


def compute_loss(log_probs, targets):
    """
    Compute negative log-likelihood loss.
    
    Args:
        log_probs: [batch, seq_len]
        targets: [batch, seq_len]
    """
    # TODO: Gather log probabilities of target indices
    target_log_probs = None 
    
    # TODO: Calculate negative mean
    loss = None 
    
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """Standard training loop for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        
        # TODO: Forward pass (with teacher forcing)
        lengths = torch.full((batch_size,), set_size, device=device)
        pointers, log_probs, _ = None, None, None 
        
        # TODO: Compute and backprop loss
        loss = None 
        
        total_loss += loss.item()
        correct += (pointers == targets).all(dim=1).sum().item()
        total += batch_size
    
    return total_loss / len(dataloader), correct / total


def demo_sorting():
    print("Training Pointer Network for Sorting")
    print("-" * 30)
    
    set_size = 5
    hidden_dim = 64
    num_epochs = 20
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # TODO: Initialize dataset, loader, model, and optimizer
    train_loader = None 
    model = None 
    optimizer = None 
    
    for epoch in range(1, num_epochs + 1):
        # TODO: Implement epoch execution
        train_loss, train_acc = 0.0, 0.0
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")


if __name__ == "__main__":
    demo_sorting()
    
    print("Exercise 3 Summary")
    print("-" * 30)
    print("""
Key concepts covered:
1. Supervised Learning for Algorithms: Learning a logical operation from data.
2. Teacher Forcing: Providing ground truth during the sequential decoding phase.
3. Exact Match Accuracy: Measuring whether the entire sequence is correctly sorted.
    """)
