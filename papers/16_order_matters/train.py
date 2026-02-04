"""
Training script for Pointer Networks on set-to-sequence tasks.

Supports multiple tasks:
- Sorting: Learn to sort numbers
- Convex Hull: Find boundary points of 2D point sets
- TSP: Traveling Salesman Problem (approximate solver)

Usage:
    python train.py --task sort --set-size 10 --epochs 100
    python train.py --task convex_hull --set-size 20 --epochs 200
    python train.py --task tsp --set-size 10 --epochs 300
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from scipy.spatial import ConvexHull as ScipyConvexHull

from implementation import ReadProcessWrite, compute_loss, accuracy


# ==============================================================================
# DATASET GENERATORS
# ==============================================================================

class SortingDataset(Dataset):
    """
    Generate random sequences to sort.
    
    Example:
        Input: [5.2, 1.8, 9.3, 2.1]
        Output pointers: [1, 3, 0, 2]  (points to indices in sorted order)
        Decoded: [1.8, 2.1, 5.2, 9.3]
    """
    
    def __init__(self, num_samples: int, set_size: int, min_val: float = 0.0, max_val: float = 10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random numbers
        values = torch.rand(self.set_size) * (self.max_val - self.min_val) + self.min_val
        
        # Ground truth: indices that would sort the array
        sorted_values, sorted_indices = torch.sort(values)
        
        # Format as [set_size, 1] for network input
        inputs = values.unsqueeze(1)
        
        return inputs, sorted_indices


class ConvexHullDataset(Dataset):
    """
    Generate random 2D point sets and compute convex hull.
    
    The convex hull is the smallest convex polygon containing all points.
    Think of it like stretching a rubber band around all the points!
    
    Example:
        Input: [(1,1), (2,3), (4,2), (3,4), (2,2)]
        Output: [(1,1), (2,3), (3,4), (4,2)]  (boundary points in order)
    """
    
    def __init__(self, num_samples: int, set_size: int):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random 2D points
        points = torch.rand(self.set_size, 2)
        
        # Compute convex hull using scipy
        try:
            hull = ScipyConvexHull(points.numpy())
            hull_indices = hull.vertices  # Indices of boundary points in order
            
            # Pad if hull has fewer points than set_size
            if len(hull_indices) < self.set_size:
                # Repeat first point to fill
                padding = [hull_indices[0]] * (self.set_size - len(hull_indices))
                hull_indices = np.concatenate([hull_indices, padding])
            
            hull_indices = torch.from_numpy(hull_indices).long()
        except Exception:
            # Degenerate case: return sorted by x-coordinate
            sorted_indices = torch.argsort(points[:, 0])
            hull_indices = sorted_indices
        
        return points, hull_indices


class TSPDataset(Dataset):
    """
    Generate random TSP instances.
    
    The Traveling Salesman Problem: visit all cities in shortest tour.
    This is NP-hard, so we use greedy nearest-neighbor as ground truth
    (not optimal, but good enough for training).
    
    Example:
        Input: [(0,0), (1,3), (4,1), (2,5)]  (city coordinates)
        Output: [0, 2, 1, 3, 0]  (tour visiting all cities)
    """
    
    def __init__(self, num_samples: int, num_cities: int):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random city coordinates [0, 1]^2
        cities = torch.rand(self.num_cities, 2)
        
        # Greedy nearest-neighbor tour (not optimal, but reasonable)
        tour = self._greedy_tour(cities)
        
        return cities, tour
    
    def _greedy_tour(self, cities):
        """Greedy nearest-neighbor TSP heuristic."""
        n = len(cities)
        unvisited = set(range(n))
        
        # Start from city 0
        current = 0
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return torch.tensor(tour, dtype=torch.long)


def get_dataset(task: str, num_samples: int, set_size: int):
    """Factory function to create dataset for a task."""
    if task == "sort":
        return SortingDataset(num_samples, set_size)
    elif task == "convex_hull":
        return ConvexHullDataset(num_samples, set_size)
    elif task == "tsp":
        return TSPDataset(num_samples, set_size)
    else:
        raise ValueError(f"Unknown task: {task}")


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device, use_teacher_forcing=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    progress = tqdm(dataloader, desc="Training")
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        
        # Forward pass
        lengths = torch.full((batch_size,), set_size, device=device)
        
        if use_teacher_forcing:
            pointers, log_probs, _ = model(
                inputs, lengths, max_steps=set_size,
                teacher_forcing=targets
            )
        else:
            pointers, log_probs, _ = model(
                inputs, lengths, max_steps=set_size
            )
        
        # Compute loss
        loss = compute_loss(log_probs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        # Metrics
        acc = accuracy(pointers, targets)
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size, set_size = inputs.size(0), inputs.size(1)
            
            # Forward pass (no teacher forcing during eval)
            lengths = torch.full((batch_size,), set_size, device=device)
            pointers, log_probs, _ = model(inputs, lengths, max_steps=set_size)
            
            # Compute metrics
            loss = compute_loss(log_probs, targets)
            acc = accuracy(pointers, targets)
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Pointer Networks for set-to-sequence tasks")
    
    # Task settings
    parser.add_argument("--task", type=str, default="sort", 
                       choices=["sort", "convex_hull", "tsp"],
                       help="Task to train on")
    parser.add_argument("--set-size", type=int, default=10,
                       help="Size of input sets")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Number of attention heads (for set encoder)")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Data settings
    parser.add_argument("--train-samples", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=1000,
                       help="Number of validation samples")
    
    # Misc
    parser.add_argument("--use-set-encoder", action="store_true", default=True,
                       help="Use order-invariant set encoder")
    parser.add_argument("--teacher-forcing", action="store_true", default=True,
                       help="Use teacher forcing during training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    # Create datasets
    print(f"\n📊 Creating datasets for task: {args.task}")
    train_dataset = get_dataset(args.task, args.train_samples, args.set_size)
    val_dataset = get_dataset(args.task, args.val_samples, args.set_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Set size: {args.set_size}")
    
    # Create model
    input_dim = 1 if args.task == "sort" else 2  # 1D for sorting, 2D for geometric tasks
    
    model = ReadProcessWrite(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        use_set_encoder=args.use_set_encoder,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️ Model created:")
    print(f"   Parameters: {num_params:,}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Set encoder: {args.use_set_encoder}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            use_teacher_forcing=args.teacher_forcing
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = save_dir / f"{args.task}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ New best model saved! (acc: {val_acc:.3f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"{args.task}_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"🎉 Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.3f}")
    print(f"   Model saved to: {save_dir / f'{args.task}_best.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
