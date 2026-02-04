"""
Solution 4: Convex Hull Solver

Complete implementation with visualization and geometry handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from tqdm import tqdm


class ConvexHullDataset(Dataset):
    """Generate random convex hull problems."""
    
    def __init__(self, num_samples=10000, min_points=5, max_points=15):
        self.num_samples = num_samples
        self.min_points = min_points
        self.max_points = max_points
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            points: Random 2D points [num_points, 2]
            target: Indices of convex hull in counter-clockwise order
        """
        num_points = np.random.randint(self.min_points, self.max_points + 1)
        
        # Random 2D points
        points = np.random.rand(num_points, 2)
        
        # Compute convex hull
        try:
            hull = ConvexHull(points)
            # hull.vertices gives indices in counter-clockwise order
            target_indices = hull.vertices
        except:
            # If points are degenerate (all collinear), use all points
            target_indices = np.arange(num_points)
        
        # Pad target to max length with -1 (will be masked)
        target = np.full(num_points, -1, dtype=np.int64)
        target[:len(target_indices)] = target_indices
        
        points_tensor = torch.from_numpy(points).float()
        target_tensor = torch.from_numpy(target).long()
        
        return points_tensor, target_tensor, len(target_indices)


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Find max length in batch
    max_len = max(points.shape[0] for points, _, _ in batch)
    
    batch_points = []
    batch_targets = []
    batch_lengths = []
    batch_hull_lengths = []
    
    for points, target, hull_len in batch:
        seq_len = points.shape[0]
        
        # Pad points
        padded_points = torch.zeros(max_len, 2)
        padded_points[:seq_len] = points
        batch_points.append(padded_points)
        
        # Pad targets
        padded_target = torch.full((max_len,), -1, dtype=torch.long)
        padded_target[:seq_len] = target
        batch_targets.append(padded_target)
        
        batch_lengths.append(seq_len)
        batch_hull_lengths.append(hull_len)
    
    return (
        torch.stack(batch_points),
        torch.stack(batch_targets),
        torch.tensor(batch_lengths),
        torch.tensor(batch_hull_lengths)
    )


class ConvexHullPointerNetwork(nn.Module):
    """Pointer network specialized for convex hull."""
    
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Decoder: LSTM
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Pointer mechanism
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, inputs, max_output_len=None):
        """
        Args:
            inputs: [batch, seq_len, 2] - 2D points
            max_output_len: Maximum decoding steps
        Returns:
            pointers: [batch, max_output_len, seq_len]
        """
        batch_size, seq_len, _ = inputs.shape
        
        if max_output_len is None:
            max_output_len = seq_len
        
        # Encode
        encoder_outputs, (h, c) = self.encoder(inputs)
        
        # Decoder initial state
        decoder_state = (h, c)
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=inputs.device)
        
        all_pointers = []
        
        for t in range(max_output_len):
            # Decode one step
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # Compute attention
            query = self.W_query(decoder_output)  # [batch, 1, hidden]
            keys = self.W_key(encoder_outputs)     # [batch, seq_len, hidden]
            
            scores = self.v(torch.tanh(query + keys.unsqueeze(1)))
            scores = scores.squeeze(-1).squeeze(1)  # [batch, seq_len]
            
            all_pointers.append(scores)
            
            # Update decoder input
            pointer_probs = torch.softmax(scores, dim=-1)
            context = torch.bmm(pointer_probs.unsqueeze(1), encoder_outputs)
            decoder_input = context
        
        pointers = torch.stack(all_pointers, dim=1)
        
        return pointers


def compute_hull_loss(logits, targets, hull_lengths):
    """
    Compute loss only for valid hull points.
    
    Args:
        logits: [batch, max_len, seq_len]
        targets: [batch, max_len] with -1 for padding
        hull_lengths: [batch] actual hull lengths
    """
    batch_size, max_len, seq_len = logits.shape
    
    total_loss = 0
    total_count = 0
    
    for b in range(batch_size):
        hull_len = hull_lengths[b].item()
        
        if hull_len > 0:
            # Get valid logits and targets
            valid_logits = logits[b, :hull_len]  # [hull_len, seq_len]
            valid_targets = targets[b, :hull_len]  # [hull_len]
            
            # Cross-entropy
            loss = nn.functional.cross_entropy(valid_logits, valid_targets)
            total_loss += loss
            total_count += 1
    
    return total_loss / max(total_count, 1)


def compute_hull_accuracy(logits, targets, hull_lengths):
    """Compute percentage of correct hull predictions."""
    batch_size = logits.shape[0]
    correct = 0
    
    for b in range(batch_size):
        hull_len = hull_lengths[b].item()
        
        if hull_len > 0:
            predictions = torch.argmax(logits[b, :hull_len], dim=-1)
            target = targets[b, :hull_len]
            
            # Check if all predictions match (order matters!)
            if torch.equal(predictions, target):
                correct += 1
    
    return correct / batch_size


def visualize_convex_hull(points, predicted_indices, true_indices=None):
    """
    Visualize convex hull prediction.
    
    Args:
        points: [num_points, 2] numpy array
        predicted_indices: List of predicted hull indices
        true_indices: List of true hull indices (optional)
    """
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Predicted hull
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=100, alpha=0.6, label='Points')
    
    # Draw predicted hull
    hull_points = points[predicted_indices]
    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the loop
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, label='Predicted Hull')
    
    # Mark hull vertices
    for i, idx in enumerate(predicted_indices):
        plt.scatter(points[idx, 0], points[idx, 1], c='red', s=200, marker='*', zorder=5)
        plt.text(points[idx, 0], points[idx, 1], str(i), fontsize=12, 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Predicted Convex Hull')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: True hull (if provided)
    if true_indices is not None:
        plt.subplot(1, 2, 2)
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=100, alpha=0.6, label='Points')
        
        # Draw true hull
        true_hull_points = points[true_indices]
        true_hull_points = np.vstack([true_hull_points, true_hull_points[0]])
        plt.plot(true_hull_points[:, 0], true_hull_points[:, 1], 'g-', linewidth=2, label='True Hull')
        
        # Mark hull vertices
        for i, idx in enumerate(true_indices):
            plt.scatter(points[idx, 0], points[idx, 1], c='green', s=200, marker='*', zorder=5)
            plt.text(points[idx, 0], points[idx, 1], str(i), fontsize=12,
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('True Convex Hull')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_convex_hull():
    """Test the convex hull solver."""
    print("🧪 Testing Convex Hull Solver")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create small dataset for testing
    train_dataset = ConvexHullDataset(num_samples=100, min_points=5, max_points=8)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Create model
    model = ConvexHullPointerNetwork(input_dim=2, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test on one example
    print("\n📊 Testing on random points:")
    
    # Generate test points
    np.random.seed(42)
    test_points = np.random.rand(8, 2)
    hull = ConvexHull(test_points)
    true_hull_indices = hull.vertices.tolist()
    
    print(f"Points shape: {test_points.shape}")
    print(f"True hull: {len(true_hull_indices)} vertices")
    print(f"Hull indices: {true_hull_indices}")
    
    # Train for a few epochs
    print("\n🏋️ Training for 5 epochs...")
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        total_acc = 0
        
        for batch_points, batch_targets, batch_lens, hull_lens in train_loader:
            batch_points = batch_points.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward
            logits = model(batch_points)
            
            # Loss
            loss = compute_hull_loss(logits, batch_targets, hull_lens)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_acc += compute_hull_accuracy(logits, batch_targets, hull_lens)
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}")
    
    # Test on example
    print("\n📊 After training:")
    model.eval()
    
    test_points_tensor = torch.from_numpy(test_points).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(test_points_tensor, max_output_len=len(true_hull_indices))
        predictions = torch.argmax(logits[0], dim=-1).cpu().numpy()
    
    print(f"Predicted hull indices: {predictions.tolist()}")
    print(f"True hull indices:      {true_hull_indices}")
    
    # Visualize
    print("\n📈 Visualizing result...")
    visualize_convex_hull(test_points, predictions, true_hull_indices)


if __name__ == "__main__":
    test_convex_hull()
    
    print("\n" + "=" * 60)
    print("🎯 Solution 4 Summary")
    print("=" * 60)
    print("""
Key implementation details:

1. **Convex Hull Problem**
   - Input: Set of 2D points
   - Output: Subset of points forming convex hull (in order!)
   - Challenge: Variable-length output (hull size < input size)

2. **Dataset Generation**
   - Use scipy.spatial.ConvexHull for ground truth
   - Returns vertices in counter-clockwise order
   - Handle degenerate cases (collinear points)

3. **Variable-Length Handling**
   - Pad targets with -1
   - Custom collate_fn for batching
   - Loss only on valid hull points

4. **Loss Computation**
   - Only compute loss for actual hull points
   - Ignore padding (-1 targets)
   - Average over batch

5. **Visualization**
   - Plot all points
   - Draw predicted hull polygon
   - Compare with true hull
   - Number vertices to show order

6. **Training Tips**
   - Start with small hulls (5-8 points)
   - Geometric problems are harder than sorting
   - May need more epochs (50-100)
   - Learning rate: 1e-3 → 1e-4 after plateau

7. **Evaluation Metrics**
   - Exact match accuracy (all indices correct)
   - Partial credit: Intersection over Union (IoU)
   - Order matters! [0,1,2] ≠ [1,2,0]

Why convex hull is harder than sorting:
- Spatial reasoning required
- Variable output length
- Geometric constraints (points must form convex polygon)
- Small errors cascade (wrong first point → all wrong)

Expected performance:
- Random guessing: ~0% (too many combinations)
- After 50 epochs: ~30-50% exact match
- After 200 epochs: ~70-80% exact match
    """)
