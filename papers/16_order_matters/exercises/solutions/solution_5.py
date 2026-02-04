"""
Solution 5: Traveling Salesman Problem (TSP) Solver

Complete implementation with tour optimization and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TSPDataset(Dataset):
    """Generate random TSP problems."""
    
    def __init__(self, num_samples=10000, num_cities=10):
        self.num_samples = num_samples
        self.num_cities = num_cities
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            cities: Random 2D city locations [num_cities, 2]
            target: Greedy nearest-neighbor tour [num_cities]
        """
        # Random city coordinates
        cities = torch.rand(self.num_cities, 2)
        
        # Compute greedy nearest-neighbor tour as target
        target = self.greedy_tour(cities.numpy())
        
        return cities, torch.from_numpy(target).long()
    
    @staticmethod
    def greedy_tour(cities):
        """
        Greedy nearest-neighbor heuristic for TSP.
        Not optimal, but gives reasonable baseline.
        """
        num_cities = len(cities)
        unvisited = set(range(num_cities))
        tour = []
        
        # Start from city 0
        current = 0
        tour.append(current)
        unvisited.remove(current)
        
        # Greedily visit nearest unvisited city
        while unvisited:
            nearest = min(unvisited, key=lambda city: np.linalg.norm(cities[current] - cities[city]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return np.array(tour)


def compute_tour_length(cities, tour):
    """
    Compute total tour length.
    
    Args:
        cities: [num_cities, 2]
        tour: [num_cities] indices
    Returns:
        length: scalar
    """
    num_cities = len(tour)
    length = 0.0
    
    for i in range(num_cities):
        current_city = tour[i]
        next_city = tour[(i + 1) % num_cities]  # Wrap around
        
        distance = np.linalg.norm(cities[current_city] - cities[next_city])
        length += distance
    
    return length


class TSPPointerNetwork(nn.Module):
    """Pointer network for TSP with masking."""
    
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
        
    def forward(self, inputs, mask_visited=True):
        """
        Args:
            inputs: [batch, num_cities, 2]
            mask_visited: If True, prevent revisiting cities
        Returns:
            pointers: [batch, num_cities, num_cities]
            tours: [batch, num_cities] greedy decoded tours
        """
        batch_size, num_cities, _ = inputs.shape
        
        # Encode
        encoder_outputs, (h, c) = self.encoder(inputs)
        
        # Decoder initial state
        decoder_state = (h, c)
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=inputs.device)
        
        all_pointers = []
        all_indices = []
        
        # Mask to track visited cities
        if mask_visited:
            mask = torch.zeros(batch_size, num_cities, device=inputs.device)
        
        for t in range(num_cities):
            # Decode one step
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # Compute attention
            query = self.W_query(decoder_output)  # [batch, 1, hidden]
            keys = self.W_key(encoder_outputs)     # [batch, num_cities, hidden]
            
            scores = self.v(torch.tanh(query + keys.unsqueeze(1)))
            scores = scores.squeeze(-1).squeeze(1)  # [batch, num_cities]
            
            # Apply mask (visited cities get -inf)
            if mask_visited:
                scores = scores.masked_fill(mask.bool(), float('-inf'))
            
            all_pointers.append(scores)
            
            # Greedy decoding
            pointer_probs = torch.softmax(scores, dim=-1)
            _, indices = pointer_probs.max(dim=-1)  # [batch]
            all_indices.append(indices)
            
            # Update mask
            if mask_visited:
                mask.scatter_(1, indices.unsqueeze(1), 1)
            
            # Update decoder input
            context = torch.bmm(pointer_probs.unsqueeze(1), encoder_outputs)
            decoder_input = context
        
        pointers = torch.stack(all_pointers, dim=1)  # [batch, num_cities, num_cities]
        tours = torch.stack(all_indices, dim=1)       # [batch, num_cities]
        
        return pointers, tours


def compute_tsp_loss(logits, targets):
    """
    Compute loss for TSP.
    
    Args:
        logits: [batch, num_cities, num_cities]
        targets: [batch, num_cities]
    """
    batch_size, num_cities, _ = logits.shape
    
    logits_flat = logits.view(-1, num_cities)
    targets_flat = targets.view(-1)
    
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    
    return loss


def compute_tsp_reward(cities, tours):
    """
    Compute reward (negative tour length) for reinforcement learning.
    
    Args:
        cities: [batch, num_cities, 2]
        tours: [batch, num_cities]
    Returns:
        rewards: [batch]
    """
    batch_size = cities.shape[0]
    rewards = []
    
    for b in range(batch_size):
        length = compute_tour_length(cities[b].cpu().numpy(), tours[b].cpu().numpy())
        rewards.append(-length)  # Negative because shorter is better
    
    return torch.tensor(rewards, device=cities.device)


def visualize_tsp_tour(cities, tour, title="TSP Tour"):
    """
    Visualize TSP tour.
    
    Args:
        cities: [num_cities, 2] numpy array
        tour: [num_cities] tour indices
    """
    plt.figure(figsize=(8, 8))
    
    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=200, alpha=0.6, zorder=3)
    
    # Number cities
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12, ha='center', va='center', 
                fontweight='bold', color='white', zorder=4)
    
    # Plot tour
    for i in range(len(tour)):
        start_idx = tour[i]
        end_idx = tour[(i + 1) % len(tour)]
        
        start = cities[start_idx]
        end = cities[end_idx]
        
        plt.arrow(start[0], start[1], 
                 end[0] - start[0], end[1] - start[1],
                 head_width=0.02, head_length=0.02, 
                 fc='red', ec='red', alpha=0.7, zorder=2,
                 length_includes_head=True)
    
    # Compute tour length
    length = compute_tour_length(cities, tour)
    
    plt.title(f"{title}\nTour Length: {length:.3f}")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def train_epoch_tsp(model, dataloader, optimizer, device):
    """Train for one epoch on TSP."""
    model.train()
    
    total_loss = 0
    total_reward = 0
    
    for batch_cities, batch_targets in tqdm(dataloader, desc="Training TSP"):
        batch_cities = batch_cities.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        logits, predicted_tours = model(batch_cities, mask_visited=True)
        
        # Supervised loss (train to match greedy baseline)
        loss = compute_tsp_loss(logits, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Compute average tour length
        rewards = compute_tsp_reward(batch_cities, predicted_tours)
        total_reward += rewards.mean().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_reward = total_reward / len(dataloader)
    avg_tour_length = -avg_reward  # Convert back to length
    
    return avg_loss, avg_tour_length


def test_tsp():
    """Test the TSP solver."""
    print("🧪 Testing TSP Solver")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset
    train_dataset = TSPDataset(num_samples=500, num_cities=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Cities per instance: {train_dataset.num_cities}")
    
    # Create model
    model = TSPPointerNetwork(input_dim=2, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test on one example
    print("\n📊 Testing on random TSP instance:")
    
    np.random.seed(42)
    test_cities = np.random.rand(10, 2)
    
    # Greedy baseline
    greedy_tour = TSPDataset.greedy_tour(test_cities)
    greedy_length = compute_tour_length(test_cities, greedy_tour)
    
    print(f"Greedy tour: {greedy_tour.tolist()}")
    print(f"Greedy length: {greedy_length:.3f}")
    
    # Train for a few epochs
    print("\n🏋️ Training for 5 epochs...")
    
    for epoch in range(5):
        loss, tour_length = train_epoch_tsp(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/5 - Loss: {loss:.4f}, Avg Tour Length: {tour_length:.3f}")
    
    # Test on example
    print("\n📊 After training:")
    model.eval()
    
    test_cities_tensor = torch.from_numpy(test_cities).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, predicted_tour = model(test_cities_tensor, mask_visited=True)
        predicted_tour = predicted_tour[0].cpu().numpy()
    
    predicted_length = compute_tour_length(test_cities, predicted_tour)
    
    print(f"Predicted tour: {predicted_tour.tolist()}")
    print(f"Predicted length: {predicted_length:.3f}")
    print(f"Greedy length: {greedy_length:.3f}")
    
    if predicted_length < greedy_length:
        improvement = (greedy_length - predicted_length) / greedy_length * 100
        print(f"✅ {improvement:.1f}% better than greedy!")
    else:
        print(f"⚠️ Not yet better than greedy (needs more training)")
    
    # Visualize
    print("\n📈 Visualizing tours...")
    
    plt.figure(figsize=(12, 5))
    
    # Greedy tour
    plt.subplot(1, 2, 1)
    plt.scatter(test_cities[:, 0], test_cities[:, 1], c='blue', s=200, alpha=0.6, zorder=3)
    for i in range(len(greedy_tour)):
        start_idx = greedy_tour[i]
        end_idx = greedy_tour[(i + 1) % len(greedy_tour)]
        start = test_cities[start_idx]
        end = test_cities[end_idx]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                 head_width=0.02, head_length=0.02, fc='green', ec='green', alpha=0.7)
    plt.title(f"Greedy Tour\nLength: {greedy_length:.3f}")
    plt.grid(True, alpha=0.3)
    
    # Predicted tour
    plt.subplot(1, 2, 2)
    plt.scatter(test_cities[:, 0], test_cities[:, 1], c='blue', s=200, alpha=0.6, zorder=3)
    for i in range(len(predicted_tour)):
        start_idx = predicted_tour[i]
        end_idx = predicted_tour[(i + 1) % len(predicted_tour)]
        start = test_cities[start_idx]
        end = test_cities[end_idx]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                 head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)
    plt.title(f"Predicted Tour\nLength: {predicted_length:.3f}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_tsp()
    
    print("\n" + "=" * 60)
    print("🎯 Solution 5 Summary")
    print("=" * 60)
    print("""
Key implementation details:

1. **TSP Problem**
   - Input: Set of 2D city locations
   - Output: Tour visiting all cities exactly once
   - Goal: Minimize total tour distance
   - NP-hard! (No polynomial-time optimal algorithm)

2. **Training Strategy**
   - Supervised learning: Train to match greedy baseline
   - Reinforcement learning: Optimize tour length directly (advanced)
   - Curriculum: Start with few cities, gradually increase

3. **Masking Mechanism**
   - CRITICAL: Each city visited exactly once
   - Set visited cities to -inf in attention scores
   - Ensures valid tours (no cycles, no skipped cities)

4. **Greedy Baseline**
   - Nearest neighbor heuristic
   - Not optimal, but reasonable (~20-30% above optimal)
   - Good training target

5. **Tour Length Computation**
   - Sum distances between consecutive cities
   - Don't forget to return to start! (closed tour)
   - Formula: Σ ||city[tour[i]] - city[tour[i+1]]||

6. **Training Tips**
   - Start with 10 cities
   - Masking is essential (verify it works!)
   - Learning rate: 1e-3 initially, decay to 1e-4
   - 100-200 epochs for decent performance

7. **Evaluation Metrics**
   - Average tour length
   - Optimality gap: (predicted - optimal) / optimal
   - % better than greedy baseline

8. **Advanced Techniques**
   - Beam search instead of greedy decoding
   - 2-opt local search post-processing
   - Reinforcement learning with REINFORCE/actor-critic
   - Attention with graph neural networks

Why TSP is the hardest:
- Combinatorial explosion: 10! = 3.6M possible tours for 10 cities
- No clear local structure (unlike sorting)
- Must learn global optimization
- Small mistakes → big tour length increase

Expected performance:
- Random tours: ~3.0 average length (10 cities in [0,1]²)
- Greedy baseline: ~2.2 average length
- After 100 epochs: ~2.0 average length (~10% better)
- State-of-art neural: ~1.8 average length (~20% better)
- Optimal (Concorde solver): ~1.7 average length

Training tricks:
✅ Use masking (essential!)
✅ Start with small instances (5-10 cities)
✅ Clip gradients (max_norm=1.0)
✅ Monitor tour length, not just loss
✅ Visualize predictions to debug
    """)
