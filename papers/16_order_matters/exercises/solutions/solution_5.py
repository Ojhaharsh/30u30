"""
Solution 5: Traveling Salesman Problem (TSP)
============================================

Deep learning solution for approximately solving the TSP.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from implementation import ReadProcessWrite


class TSPDataset(Dataset):
    def __init__(self, num_samples, num_cities):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cities = torch.rand(self.num_cities, 2)
        unvisited = set(range(self.num_cities))
        current = 0
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return cities, torch.tensor(tour, dtype=torch.long)


def train_tsp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReadProcessWrite(input_dim=2, hidden_dim=128, use_set_encoder=True, num_layers=3).to(device)
    dataset = TSPDataset(2000, 10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for cities, targets in loader:
        cities, targets = cities.to(device), targets.to(device)
        lengths = torch.full((cities.size(0),), 10, device=device)
        _, log_probs, _ = model(cities, lengths, 10, teacher_forcing=targets)
        
        loss = -log_probs.gather(1, targets).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    print("TSP solution loaded.")
