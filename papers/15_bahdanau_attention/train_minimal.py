"""
Day 15: Training Script for Bahdanau Attention Seq2Seq

Orchestrates the training of a Sequence-to-Sequence model on a number reversal task.
Uses Bahdanau (additive) attention for alignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
import argparse
from implementation import create_model

class ReversalDataset(Dataset):
    """
    Synthetic dataset for sequence reversal.
    Input: [5, 3, 8] -> Target: [8, 3, 5]
    """
    def __init__(self, num_samples=10000, min_len=5, max_len=15, vocab_size=100):
        self.samples = []
        self.vocab_size = vocab_size
        self.pad_idx, self.sos_idx, self.eos_idx = 0, 1, 2
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            src = [random.randint(3, vocab_size - 1) for _ in range(length)]
            trg = list(reversed(src))
            self.samples.append((src, trg))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        src, trg = self.samples[idx]
        trg = [self.sos_idx] + trg + [self.eos_idx]
        return torch.tensor(src), torch.tensor(trg)

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in srcs])
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    trg_padded = nn.utils.rnn.pad_sequence(trgs, batch_first=True, padding_value=0)
    return src_padded, src_lengths, trg_padded

def train_epoch(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    for src, lengths, trg in loader:
        src, lengths, trg = src.to(device), lengths.to(device), trg.to(device)
        optimizer.zero_grad()
        
        outputs, _ = model(src, lengths, trg)
        output_dim = outputs.shape[-1]
        
        loss = criterion(outputs.reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, lengths, trg in loader:
            src, lengths, trg = src.to(device), lengths.to(device), trg.to(device)
            outputs, _ = model(src, lengths, trg)
            output_dim = outputs.shape[-1]
            loss = criterion(outputs.reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Day 15: Bahdanau Attention Training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="best_model.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_data = ReversalDataset(num_samples=10000, vocab_size=args.vocab_size)
    val_data = ReversalDataset(num_samples=1000, vocab_size=args.vocab_size)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
    
    model = create_model(args.vocab_size, args.vocab_size, hidden_dim=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-t0:.1f}s")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            
    print(f"\nTraining Complete. Best model saved to {args.save_path}")

if __name__ == '__main__':
    main()
