"""
Exercise 4: Convex Hull with Pointer Networks

Learn to solve geometric problems with neural networks!

Task: Given a set of 2D points, find the boundary points (convex hull)
      in clockwise or counterclockwise order.

Real-world analogy: Imagine points are cities. The convex hull is like
                   stretching a rubber band around them - it touches
                   only the outermost cities!

Geometric problem + Neural network = 🤯

Your task: Train a model to learn convex hull geometry.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial import ConvexHull as ScipyConvexHull
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from implementation import ReadProcessWrite


class ConvexHullDataset(Dataset):
    """
    Generate random point sets and compute their convex hulls.
    
    The convex hull of a set of points is the smallest convex polygon
    that contains all the points.
    """
    
    def __init__(self, num_samples, num_points):
        self.num_samples = num_samples
        self.num_points = num_points
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # TODO: Generate random 2D points in [0, 1]^2
        points = None  # TODO: torch.rand(self.num_points, 2)
        
        # TODO: Compute convex hull using scipy
        try:
            hull = ScipyConvexHull(points.numpy())
            hull_indices = hull.vertices  # Boundary point indices
            
            # TODO: Pad if hull has fewer points than num_points
            # If convex hull has 6 points but we need 10, repeat first point
            if len(hull_indices) < self.num_points:
                padding = [hull_indices[0]] * (self.num_points - len(hull_indices))
                hull_indices = np.concatenate([hull_indices, padding])
            
            hull_indices = torch.from_numpy(hull_indices).long()
        except:
            # Degenerate case (collinear points): return sorted by x
            hull_indices = torch.argsort(points[:, 0])
        
        return points, hull_indices


def visualize_prediction(points, predicted_hull, true_hull):
    """
    Visualize the convex hull prediction.
    
    This helps us SEE if the model learned geometry!
    """
    points = points.cpu().numpy()
    predicted_hull = predicted_hull.cpu().numpy()
    true_hull = true_hull.cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # True convex hull
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=100, alpha=0.6, label='Points')
    
    # Draw true hull
    hull_points = points[true_hull]
    # Close the polygon by adding first point at end
    hull_points_closed = np.vstack([hull_points, hull_points[0]])
    plt.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 
             'g-', linewidth=2, label='True Hull')
    plt.scatter(hull_points[:, 0], hull_points[:, 1], c='green', s=200, marker='*')
    
    plt.title('Ground Truth Convex Hull')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Predicted convex hull
    plt.subplot(1, 2, 2)
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=100, alpha=0.6, label='Points')
    
    # Draw predicted hull
    pred_hull_points = points[predicted_hull]
    pred_hull_points_closed = np.vstack([pred_hull_points, pred_hull_points[0]])
    plt.plot(pred_hull_points_closed[:, 0], pred_hull_points_closed[:, 1], 
             'r--', linewidth=2, label='Predicted Hull')
    plt.scatter(pred_hull_points[:, 0], pred_hull_points[:, 1], c='red', s=200, marker='*')
    
    plt.title('Model Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('convex_hull_prediction.png', dpi=150)
    print("💾 Saved visualization to convex_hull_prediction.png")


def compute_hull_accuracy(predicted, true, points):
    """
    Compute accuracy for convex hull.
    
    This is tricky because:
    1. The hull might have fewer points than the set
    2. We care about whether predicted points are actually on the boundary
    """
    # TODO: Check if predicted indices match true indices (exact match)
    exact_match = None  # TODO: (predicted == true).all().item()
    
    # TODO: Check if predicted points form a valid convex hull
    # Hint: Check if predicted points are a subset of true hull points
    pred_set = set(predicted.tolist())
    true_set = set(true.tolist()[:len(predicted)])  # Compare same length
    
    overlap = len(pred_set.intersection(true_set)) / len(true_set)
    
    return exact_match, overlap


def demo_convex_hull():
    """
    Train a pointer network to solve convex hull!
    
    This is mind-blowing: The model learns GEOMETRY through examples!
    No hard-coded algorithms, just learning from data.
    """
    print("🎯 Training Pointer Network for Convex Hull")
    print("=" * 60)
    
    # Hyperparameters
    num_points = 10       # 10 points per set
    hidden_dim = 128
    num_epochs = 50
    batch_size = 64
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Create datasets
    train_dataset = ConvexHullDataset(2000, num_points)
    val_dataset = ConvexHullDataset(400, num_points)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    # TODO: Create model with set encoder
    # This is key: We want order-invariant encoding!
    model = ReadProcessWrite(
        input_dim=2,  # 2D points (x, y)
        hidden_dim=hidden_dim,
        use_set_encoder=True,  # Order doesn't matter for input!
        num_heads=4,
        num_layers=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🏗️ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training loop
    print("\n🚀 Training...")
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        train_exact = 0
        train_overlap = 0
        
        for points, hull_indices in train_loader:
            points = points.to(device)
            hull_indices = hull_indices.to(device)
            batch_size = points.size(0)
            
            # Forward
            lengths = torch.full((batch_size,), num_points, device=device)
            pred_hull, log_probs, _ = model(
                points, lengths, num_points,
                teacher_forcing=hull_indices
            )
            
            # Loss (negative log-likelihood)
            target_log_probs = log_probs.gather(1, hull_indices)
            loss = -target_log_probs.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            for i in range(batch_size):
                exact, overlap = compute_hull_accuracy(
                    pred_hull[i], hull_indices[i], points[i]
                )
                train_exact += exact
                train_overlap += overlap
        
        # Validation
        model.eval()
        val_exact = 0
        val_overlap = 0
        val_samples = 0
        
        with torch.no_grad():
            for points, hull_indices in val_loader:
                points = points.to(device)
                hull_indices = hull_indices.to(device)
                batch_size = points.size(0)
                
                lengths = torch.full((batch_size,), num_points, device=device)
                pred_hull, _, _ = model(points, lengths, num_points)
                
                for i in range(batch_size):
                    exact, overlap = compute_hull_accuracy(
                        pred_hull[i], hull_indices[i], points[i]
                    )
                    val_exact += exact
                    val_overlap += overlap
                val_samples += batch_size
        
        train_loss /= len(train_loader)
        train_exact /= len(train_dataset)
        train_overlap /= len(train_dataset)
        val_exact /= val_samples
        val_overlap /= val_samples
        
        if val_overlap > best_acc:
            best_acc = val_overlap
            status = "🌟"
        else:
            status = ""
        
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
              f"Val Exact: {val_exact:.3f} | Val Overlap: {val_overlap:.3f} {status}")
    
    # Visualize a prediction
    print("\n" + "=" * 60)
    print("🎨 Visualizing prediction...")
    
    model.eval()
    with torch.no_grad():
        # Get a test example
        points, true_hull = val_dataset[0]
        points = points.unsqueeze(0).to(device)
        lengths = torch.tensor([num_points]).to(device)
        
        pred_hull, _, _ = model(points, lengths, num_points)
        
        visualize_prediction(points[0], pred_hull[0], true_hull)


if __name__ == "__main__":
    demo_convex_hull()
    
    print("\n" + "=" * 60)
    print("🎯 Exercise 4 Summary")
    print("=" * 60)
    print("""
You've taught a neural network GEOMETRY! 🤯

Key concepts you learned:
1. ✅ Neural networks can learn geometric reasoning
2. ✅ Set encoder is crucial (input point order doesn't matter)
3. ✅ Output order DOES matter (boundary must be traced in sequence)
4. ✅ Evaluation metrics for geometric problems
5. ✅ Visualization helps debug and understand the model

Observations:
- The model learns which points are on the boundary
- It learns to trace the boundary in order (clockwise/counterclockwise)
- No hard-coded geometry algorithms - pure learning!
- Generalizes to unseen point configurations

Next: Exercise 5 - The ultimate challenge: Traveling Salesman Problem!
    """)
