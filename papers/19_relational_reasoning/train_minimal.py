"""
train_minimal.py - Relational Reasoning Training & Generalization Suite

This script provides a research-grade CLI for training Relation Networks. 
New to this version is the "Set Size Generalization" evaluation.

Key Features:
1. Multi-Aggregator Comparison: Training with sum, mean, or max.
2. Set Size Generalization: Train on N objects, test on M objects.
3. Coordinate Injection: Testing the effect of spatial awareness.
4. Robust Performance Metrics: Tracking loss and accuracy across variants.

Usage:
    # Train with different aggregators
    python train_minimal.py --aggregator mean --epochs 20
    
    # Run generalization test (Train N=5, Test N=15)
    python train_minimal.py --train-n 5 --test-n 15 --epochs 30

Reference: Santoro et al. (2017) - Section 4.2 (Generalization)
Author: 30u30 Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from typing import Tuple, Dict

from implementation import RelationNetwork, add_coordinates

# ============================================================================
# RELATIONAL DATASETS
# ============================================================================

class RelationalDataset(Dataset):
    """
    Datasets for relational reasoning tasks.
    Supports 'furthest' (spatial task) and 'count' (cardinality task).
    """
    def __init__(
        self, 
        mode: str = 'furthest', 
        num_samples: int = 5000, 
        num_objects: int = 10,
        use_coords: bool = False
    ):
        self.mode = mode
        self.use_coords = use_coords
        self.samples = []
        
        for _ in range(num_samples):
            # Objects are 2D coordinates in [-1, 1]
            points = np.random.uniform(-1, 1, (num_objects, 2))
            
            if mode == 'furthest':
                # Label: Index of the point furthest from origin
                # Proxies for "Behind/Front" spatial reasoning in CLEVR (Section 4.1).
                distances = np.linalg.norm(points, axis=1)
                target = np.argmax(distances)
            elif mode == 'count':
                # Label: Count points in top-right quadrant
                # Proxies for "How many" counting tasks in CLEVR (Section 4.1).
                # Explicitly tests the 'sum' vs 'mean' inductive bias.
                target = np.sum((points[:, 0] > 0) & (points[:, 1] > 0))
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            self.samples.append((points, target))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        points, target = self.samples[idx]
        points_tensor = torch.tensor(points).float()
        
        # If the model is coordinate-aware, we use the specific utility
        if self.use_coords:
            points_tensor = add_coordinates(points_tensor.unsqueeze(0)).squeeze(0)
            
        return points_tensor, torch.tensor(target).long()


# ============================================================================
# TRAINING LOGIC
# ============================================================================

def run_evaluation(model, loader, device):
    """Utility to compute accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_experiment(args):
    """Main experiment loop covering training and generalization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Day 19 Experiment: [Task: {args.mode}] ---")
    print(f"Aggregator: {args.aggregator} | Train N: {args.train_n} | Test N: {args.test_n}")
    
    # 1. Prepare Training Data (N objects)
    train_dataset = RelationalDataset(
        mode=args.mode, 
        num_samples=args.train_samples, 
        num_objects=args.train_n,
        use_coords=args.use_coords
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Prepare Testing Data (M objects - Generalization Test)
    # If test_n != train_n, this tests if the RN's logic generalizes to larger sets.
    test_dataset = RelationalDataset(
        mode=args.mode, 
        num_samples=1000, 
        num_objects=args.test_n,
        use_coords=args.use_coords
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 3. Build Model
    # Determine output dimensions
    if args.mode == 'furthest':
        # Labels are 0 to N-1 for train, but for test N, it might be 0 to M-1.
        # This is a key limitation: the final f_phi MLP is fixed to a certain output size.
        # For generalization across set sizes, the task must have a fixed output space 
        # (like 'count' or 'binary comparison').
        if args.train_n != args.test_n:
            print("[WARNING] Set size generalization on 'furthest' task is limited by f_phi output neurons.")
        num_classes = max(args.train_n, args.test_n)
    else:
        # Counting task: output space is 0 to num_objects
        num_classes = args.test_n + 1
        
    model = RelationNetwork(
        object_dim=4 if args.use_coords else 2, 
        relation_dim=args.hidden_dim, 
        output_dim=num_classes,
        aggregator=args.aggregator
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            # In counting, labels are indices 0 to count.
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            train_acc = run_evaluation(model, train_loader, device)
            test_acc = run_evaluation(model, test_loader, device)
            print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Train Acc (N={args.train_n}): {train_acc:.3f} | Test Acc (N={args.test_n}): {test_acc:.3f}")

    print("-" * 50)
    print(f"Experiment Complete. Generalization Gap: {abs(train_acc - test_acc):.3f}")
    
    if args.save_model:
        torch.save(model.state_dict(), f"rn_{args.mode}_{args.aggregator}.pt")
        
    return model

# ============================================================================
# CLI CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Day 19: Relation Network Generalization Suite")
    
    # General Parameters
    parser.add_argument('--mode', type=str, default='count', choices=['furthest', 'count'],
                        help='Task mode (counting is better for generalization tests)')
    parser.add_argument('--aggregator', type=str, default='sum', choices=['sum', 'mean', 'max'],
                        help='Pooling strategy for relations (sum is standard)')
    parser.add_argument('--use-coords', action='store_true', help='Inject (x, y) coordinates into objects')
    
    # Set Size Parameters
    parser.add_argument('--train-n', type=int, default=10, help='Set size during training')
    parser.add_argument('--test-n', type=int, default=10, help='Set size during testing (Generalization)')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--train-samples', type=int, default=10000)
    
    parser.add_argument('--save-model', action='store_true')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
