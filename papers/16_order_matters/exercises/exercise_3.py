"""
Exercise 3: Training Pointer Networks for Sorting

Learn to train a model to sort numbers using pointer networks.

Task: Given a random list like [5.2, 1.8, 9.3, 2.1], output pointers
      [1, 3, 0, 2] which corresponds to sorted order [1.8, 2.1, 5.2, 9.3]

Real-world analogy: Teaching a robot to arrange books by height without
                   explicitly programming the sorting algorithm!

Your task: Implement the training loop and loss function.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from implementation import PointerNetwork


class SortingDataset(Dataset):
    """Generate random sorting problems."""
    
    def __init__(self, num_samples, set_size, min_val=0.0, max_val=10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # TODO: Generate random numbers
        values = None  # TODO: torch.rand(self.set_size) * (max - min) + min
        
        # TODO: Get sorted indices (ground truth)
        # Hint: torch.sort returns (sorted_values, sorted_indices)
        sorted_values, target = None, None  # TODO
        
        # Format as [set_size, 1] for network
        inputs = None  # TODO: values.unsqueeze(1)
        
        return inputs, target


def compute_loss(log_probs, targets):
    """
    Compute negative log-likelihood loss.
    
    Args:
        log_probs: [batch, seq_len] - Log probabilities for each position
        targets: [batch, seq_len] - Ground truth indices
        
    Returns:
        loss: Scalar tensor
    """
    # TODO: Gather log probabilities of target indices
    # Hint: log_probs.gather(1, targets) selects log_probs[i, targets[i]]
    target_log_probs = None  # TODO
    
    # TODO: Negative log-likelihood (we want to maximize log prob = minimize negative)
    loss = None  # TODO: -target_log_probs.mean()
    
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        
        # TODO: Forward pass
        lengths = torch.full((batch_size,), set_size, device=device)
        pointers, log_probs, _ = None, None, None  # TODO: model(inputs, lengths, set_size, teacher_forcing=targets)
        
        # TODO: Compute loss
        loss = None  # TODO: compute_loss(log_probs, targets)
        
        # TODO: Backward pass
        # Hint: optimizer.zero_grad(), loss.backward(), optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        correct += (pointers == targets).all(dim=1).sum().item()
        total += batch_size
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size, set_size = inputs.size(0), inputs.size(1)
            
            # TODO: Forward pass (no teacher forcing!)
            lengths = torch.full((batch_size,), set_size, device=device)
            pointers, _, _ = None, None, None  # TODO: model(inputs, lengths, set_size)
            
            # Count exact matches
            correct += (pointers == targets).all(dim=1).sum().item()
            total += batch_size
    
    return correct / total


def demo_sorting():
    """
    Train a pointer network to sort numbers!
    
    This is like teaching someone to sort without showing them the algorithm.
    They learn by seeing examples: [5,2,9,1] → [1,2,5,9]
    """
    print("🎯 Training Pointer Network for Sorting")
    print("=" * 60)
    
    # Hyperparameters
    set_size = 5          # Sort 5 numbers
    hidden_dim = 64       # Model size
    num_epochs = 20       # Training epochs
    batch_size = 32
    lr = 1e-3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Create datasets
    train_dataset = None  # TODO: SortingDataset(1000, set_size)
    val_dataset = None    # TODO: SortingDataset(200, set_size)
    
    # TODO: Create dataloaders
    train_loader = None   # TODO: DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = None     # TODO: DataLoader(val_dataset, batch_size, shuffle=False)
    
    # TODO: Create model
    model = None  # TODO: PointerNetwork(input_dim=1, hidden_dim=hidden_dim).to(device)
    
    # TODO: Create optimizer
    optimizer = None  # TODO: optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🏗️ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📊 Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Training loop
    print("\n🚀 Training...")
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        # TODO: Train
        train_loss, train_acc = None, None  # TODO: train_epoch(model, train_loader, optimizer, device)
        
        # TODO: Evaluate
        val_acc = None  # TODO: evaluate(model, val_loader, device)
        
        if val_acc > best_acc:
            best_acc = val_acc
            status = "🌟 NEW BEST!"
        else:
            status = ""
        
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} {status}")
    
    # Test on a specific example
    print("\n" + "=" * 60)
    print("🧪 Testing on a specific example")
    
    model.eval()
    with torch.no_grad():
        # Create a test example
        test_values = torch.tensor([[5.0], [2.0], [9.0], [1.0], [7.0]])
        test_input = test_values.unsqueeze(0).to(device)
        lengths = torch.tensor([5]).to(device)
        
        pointers, log_probs, attentions = model(test_input, lengths, 5)
        
        print(f"\nInput:    {test_values.squeeze().tolist()}")
        print(f"Pointers: {pointers[0].tolist()}")
        
        # Decode pointers to sorted values
        sorted_values = test_values[pointers[0]].squeeze()
        print(f"Sorted:   {sorted_values.tolist()}")
        
        # Check if correct
        expected = torch.sort(test_values.squeeze())[0]
        is_correct = torch.allclose(sorted_values, expected, atol=1e-6)
        print(f"\n{'✅ CORRECT!' if is_correct else '❌ WRONG!'}")


if __name__ == "__main__":
    demo_sorting()
    
    print("\n" + "=" * 60)
    print("🎯 Exercise 3 Summary")
    print("=" * 60)
    print("""
You've trained a neural network to sort!

Key concepts you learned:
1. ✅ Supervised learning for combinatorial problems
2. ✅ Negative log-likelihood loss
3. ✅ Teacher forcing: using ground truth during training
4. ✅ Exact match accuracy: sequence must be 100% correct
5. ✅ The model learns the CONCEPT of sorting, not a hard-coded algorithm

Observations:
- Small sets (5-10 elements) are easy to learn
- Larger sets require more training time
- The model can generalize to unseen number ranges!

Next: Exercise 4 - Solve the convex hull problem!
    """)
