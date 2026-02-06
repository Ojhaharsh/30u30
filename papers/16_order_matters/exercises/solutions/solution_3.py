"""
Solution 3: Training Pointer Networks for Sorting
==================================================

Implementation of the training loop for numerical sorting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from implementation import PointerNetwork


class SortingDataset(Dataset):
    def __init__(self, num_samples, set_size, min_val=0.0, max_val=10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size) * (self.max_val - self.min_val) + self.min_val
        _, target = torch.sort(values)
        inputs = values.unsqueeze(1)
        return inputs, target


def compute_loss(log_probs, targets):
    # Select the log probability assigned to the ground-truth index
    target_log_probs = log_probs.gather(1, targets)
    return -target_log_probs.mean()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        
        lengths = torch.full((batch_size,), set_size, device=device)
        pointers, log_probs, _ = model(inputs, lengths, set_size, teacher_forcing=targets)
        
        loss = compute_loss(log_probs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (pointers == targets).all(dim=1).sum().item()
        total += batch_size
    
    return total_loss / len(dataloader), correct / total

if __name__ == "__main__":
    print("Sorting Training Solution loaded.")
