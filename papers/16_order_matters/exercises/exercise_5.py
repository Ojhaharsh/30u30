"""
Exercise 5: Traveling Salesman Problem with Pointer Networks

The ultimate challenge: Solve TSP with neural networks!

Task: Given a set of city coordinates, find the shortest tour that visits
      all cities exactly once and returns to the start.

Real-world: This is one of the most famous NP-hard problems!
           - Delivery route optimization
           - Circuit board drilling
           - DNA sequencing

Your task: Train a model to approximate TSP solutions.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from implementation import ReadProcessWrite


class TSPDataset(Dataset):
    """
    Generate random TSP instances.
    
    We use greedy nearest-neighbor as ground truth (not optimal, but good).
    The model might actually learn to do BETTER than greedy!
    """
    
    def __init__(self, num_samples, num_cities):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # TODO: Generate random city coordinates in [0, 1]^2
        cities = None  # TODO: torch.rand(self.num_cities, 2)
        
        # TODO: Compute greedy nearest-neighbor tour
        tour = self._greedy_tour(cities)
        
        return cities, tour
    
    def _greedy_tour(self, cities):
        """
        Greedy nearest-neighbor heuristic.
        
        Algorithm:
        1. Start at city 0
        2. Repeat: Go to nearest unvisited city
        3. Return tour
        
        This is NOT optimal, but gives ~1.25x optimal on average.
        """
        n = len(cities)
        unvisited = set(range(n))
        
        # TODO: Start from city 0
        current = 0
        tour = [current]
        unvisited.remove(current)
        
        # TODO: Greedily select nearest unvisited city
        while unvisited:
            # Find nearest city
            nearest = min(unvisited, 
                         key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return torch.tensor(tour, dtype=torch.long)


def compute_tour_length(tour, cities):
    """
    Compute total tour length (including return to start).
    
    Args:
        tour: [num_cities] - Indices of cities in visit order
        cities: [num_cities, 2] - City coordinates
        
    Returns:
        length: Total euclidean distance
    """
    # TODO: Get coordinates in tour order
    tour_cities = cities[tour]
    
    # TODO: Compute distances between consecutive cities
    # Hint: tour_cities[1:] - tour_cities[:-1] gives adjacent pairs
    distances = torch.norm(tour_cities[1:] - tour_cities[:-1], dim=1)
    
    # TODO: Add distance back to start
    return_distance = torch.norm(tour_cities[-1] - tour_cities[0])
    
    total_length = distances.sum() + return_distance
    return total_length.item()


def visualize_tour(cities, tour, title="TSP Tour"):
    """
    Visualize the TSP tour.
    
    Blue dots = cities
    Red line = tour path
    """
    cities = cities.cpu().numpy()
    tour = tour.cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    
    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=200, alpha=0.6, zorder=3)
    
    # Label cities
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12, ha='center', va='center', 
                color='white', weight='bold')
    
    # Plot tour
    tour_cities = cities[tour]
    # Close the tour by returning to start
    tour_cities_closed = np.vstack([tour_cities, tour_cities[0]])
    
    plt.plot(tour_cities_closed[:, 0], tour_cities_closed[:, 1], 
            'r-', linewidth=2, alpha=0.7, zorder=2)
    
    # Mark start city
    plt.scatter(cities[tour[0], 0], cities[tour[0], 1], 
               c='green', s=400, marker='*', zorder=4, label='Start')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()


def demo_tsp():
    """
    Train a pointer network to solve TSP!
    
    This is the holy grail of combinatorial optimization.
    Can a neural network learn to find good tours?
    
    Spoiler: YES! Not always optimal, but impressively good!
    """
    print("🎯 Training Pointer Network for TSP")
    print("=" * 60)
    print("This is an NP-hard problem - we're attempting the impossible!")
    
    # Hyperparameters
    num_cities = 10      # Start small (TSP is HARD!)
    hidden_dim = 128
    num_epochs = 100
    batch_size = 64
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # TODO: Create datasets
    train_dataset = TSPDataset(3000, num_cities)
    val_dataset = TSPDataset(500, num_cities)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    # TODO: Create model
    model = ReadProcessWrite(
        input_dim=2,  # 2D coordinates
        hidden_dim=hidden_dim,
        use_set_encoder=True,  # Cities are a set!
        num_heads=4,
        num_layers=3  # Deeper for harder problem
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"\n🏗️ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📊 Train: {len(train_dataset)} TSP instances")
    
    # Training loop
    print("\n🚀 Training...")
    best_val_length = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        
        for cities, tour in train_loader:
            cities = cities.to(device)
            tour = tour.to(device)
            batch_size = cities.size(0)
            
            # Forward
            lengths = torch.full((batch_size,), num_cities, device=device)
            pred_tour, log_probs, _ = model(
                cities, lengths, num_cities,
                teacher_forcing=tour
            )
            
            # Loss
            target_log_probs = log_probs.gather(1, tour)
            loss = -target_log_probs.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_greedy_length = 0
        val_model_length = 0
        val_samples = 0
        
        with torch.no_grad():
            for cities, greedy_tour in val_loader:
                cities = cities.to(device)
                batch_size = cities.size(0)
                
                lengths = torch.full((batch_size,), num_cities, device=device)
                pred_tour, _, _ = model(cities, lengths, num_cities)
                
                # Compare tour lengths
                for i in range(batch_size):
                    greedy_len = compute_tour_length(greedy_tour[i], cities[i])
                    model_len = compute_tour_length(pred_tour[i], cities[i])
                    
                    val_greedy_length += greedy_len
                    val_model_length += model_len
                
                val_samples += batch_size
        
        avg_greedy = val_greedy_length / val_samples
        avg_model = val_model_length / val_samples
        improvement = (avg_greedy - avg_model) / avg_greedy * 100
        
        scheduler.step(avg_model)
        
        if avg_model < best_val_length:
            best_val_length = avg_model
            status = "🌟 BEST!"
        else:
            status = ""
        
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Greedy: {avg_greedy:.4f} | Model: {avg_model:.4f} | "
              f"Improv: {improvement:+.1f}% {status}")
    
    # Visualize predictions
    print("\n" + "=" * 60)
    print("🎨 Visualizing predictions...")
    
    model.eval()
    with torch.no_grad():
        # Get 3 test examples
        for idx in range(3):
            cities, greedy_tour = val_dataset[idx]
            cities_batch = cities.unsqueeze(0).to(device)
            lengths = torch.tensor([num_cities]).to(device)
            
            pred_tour, _, _ = model(cities_batch, lengths, num_cities)
            
            greedy_len = compute_tour_length(greedy_tour, cities)
            model_len = compute_tour_length(pred_tour[0], cities)
            
            print(f"\nExample {idx + 1}:")
            print(f"  Greedy tour length: {greedy_len:.4f}")
            print(f"  Model tour length:  {model_len:.4f}")
            print(f"  {'✅ Better!' if model_len < greedy_len else '➡️ Same' if abs(model_len - greedy_len) < 0.01 else '❌ Worse'}")
            
            # Visualize
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            visualize_tour(cities, greedy_tour, f"Greedy (length: {greedy_len:.3f})")
            
            plt.subplot(1, 2, 2)
            visualize_tour(cities, pred_tour[0], f"Model (length: {model_len:.3f})")
            
            plt.savefig(f'tsp_example_{idx + 1}.png', dpi=150)
            print(f"  💾 Saved to tsp_example_{idx + 1}.png")
            plt.close()


if __name__ == "__main__":
    demo_tsp()
    
    print("\n" + "=" * 60)
    print("🎯 Exercise 5 Summary - THE FINALE!")
    print("=" * 60)
    print("""
You've trained a neural network for an NP-hard problem! 🤯🎉

Key concepts you learned:
1. ✅ Neural networks can tackle combinatorial optimization
2. ✅ Greedy algorithms as training targets (teacher forcing)
3. ✅ Tour length as a quality metric
4. ✅ The model often BEATS the greedy baseline!
5. ✅ Set encoder + Pointer decoder = powerful combo

Mind-blowing insights:
- The model learns spatial reasoning without explicit geometry
- It discovers tour optimization strategies through examples
- Sometimes finds better solutions than the greedy teacher!
- Generalizes to different city configurations

Limitations:
- Still not optimal (optimal is NP-hard)
- Struggles with very large instances (50+ cities)
- Training time grows with problem size

Real-world applications:
- Route optimization (delivery, logistics)
- Manufacturing (drill holes, weld points)
- Genome sequencing
- Circuit design

🎓 Congratulations! You've completed all 5 exercises!

You now understand:
- ✅ Pointer Networks
- ✅ Order-invariant set encoding
- ✅ Read-Process-Write framework
- ✅ When order matters and when it doesn't
- ✅ Neural approaches to combinatorial problems

Next steps:
- Try larger problem sizes
- Implement beam search for better solutions
- Read about Set Transformers (modern extension)
- Apply to your own set-to-sequence problems!
    """)
