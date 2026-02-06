"""
Minimal Training Script for Pointer Networks
Standardized for Day 18 of 30u30

Train a Pointer Network on a sorting task. The model learns to output
the indices of input numbers in increasing order.

Usage:
    python train_minimal.py --epochs 20 --seq_len 5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from implementation import PointerNetwork

class SortingDataset(Dataset):
    """
    Dataset of random number sequences and their sorted indices.
    
    Args:
        num_samples (int): Total sequences to generate.
        seq_len (int): Length of each sequence.
    """
    def __init__(self, num_samples=5000, seq_len=5):
        self.samples = []
        for _ in range(num_samples):
            # Generate random numbers in [0, 1)
            nums = np.random.rand(seq_len)
            # Targets are the indices that would sort the array
            indices = np.argsort(nums)
            self.samples.append((nums, indices))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        nums, indices = self.samples[idx]
        return torch.tensor(nums).float().unsqueeze(-1), torch.tensor(indices).long()

def train(epochs=20, seq_len=5, batch_size=64, hidden_size=64, lr=0.001):
    """
    Standard training logic for the Pointer Network.
    
    Args:
        epochs (int): Number of training passes.
        seq_len (int): Input sequence length.
        batch_size (int): Samples per update.
        hidden_size (int): Model capacity.
        lr (float): Learning rate for Adam.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === Step 1: Prepare Data ===
    train_data = SortingDataset(10000, seq_len=seq_len)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # === Step 2: Initialize Model ===
    model = PointerNetwork(input_size=1, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    print(f"\nTraining Pointer Network on '{device}'")
    print(f"Sequence Length: {seq_len} | Hidden Size: {hidden_size}")
    print("=" * 60)
    
    start_time = time.time()
    
    # === Step 3: Training Loop ===
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            
            # Compute loss (flatten batch and time dimensions)
            loss = criterion(output.view(-1, seq_len), y.view(-1))
            
            # Backward pass
            loss.backward()
            # Gradient clipping is standard for Ptr-Nets to stabilize RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    print("=" * 60)
    print(f"Training complete! Final Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "model_weights.pth")
    print("[OK] Model saved to model_weights.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pointer Network Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=5, help="Length of sorting sequence")
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs, 
        seq_len=args.seq_len, 
        hidden_size=args.hidden_size, 
        lr=args.lr
    )
