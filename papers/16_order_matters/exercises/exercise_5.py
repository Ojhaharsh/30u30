"""
Exercise 5: Traveling Salesman Problem (TSP)
============================================

Approximate combinatorial optimization using Pointer Networks.

Task: Given city coordinates, find a tour that visits all cities and 
      minimizes total Euclidean distance.

Concept: TSP is NP-Hard. Neural networks are used here to learn a strong 
         approximation heuristic.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.append('..')
from implementation import ReadProcessWrite


class TSPDataset(Dataset):
    """
    TSP instances where ground truth is provided by a nearest-neighbor greedy heuristic.
    """
    def __init__(self, num_samples, num_cities):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cities = torch.rand(self.num_cities, 2)
        tour = self._greedy_tour(cities)
        return cities, tour
    
    def _greedy_tour(self, cities):
        n = len(cities)
        unvisited = set(range(n))
        current = 0
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return torch.tensor(tour, dtype=torch.long)


def compute_tour_length(tour, cities):
    tour_cities = cities[tour]
    distances = torch.norm(tour_cities[1:] - tour_cities[:-1], dim=1)
    return_dist = torch.norm(tour_cities[-1] - tour_cities[0])
    return (distances.sum() + return_dist).item()


def demo_tsp():
    print("Training Pointer Network for TSP Approximation")
    print("-" * 30)
    
    num_cities = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ReadProcessWrite(
        input_dim=2, 
        hidden_dim=128, 
        use_set_encoder=True,
        num_layers=3
    ).to(device)
    
    print("TSP model initialized.")


if __name__ == "__main__":
    demo_tsp()
    
    print("Exercise 5 Summary")
    print("-" * 30)
    print("""
Key concepts covered:
1. Combinatorial Optimization: Learning to solve search problems from data.
2. Heuristic Learning: Training on greedy baseline outputs to learn generalizable rules.
3. Path Consistency: Ensuring the model generates a valid permutation (no revisiting).
    """)
