"""
Training script for Pointer Networks.
=====================================

Tasks supported:
- sorting: numerical sorting
- convex_hull: boundary point identification
- tsp: traveling salesman problem approximation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import ConvexHull as ScipyConvexHull

from implementation import ReadProcessWrite, compute_loss, accuracy


class SortingDataset(Dataset):
    def __init__(self, num_samples: int, set_size: int, min_val: float = 0.0, max_val: float = 10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size) * (self.max_val - self.min_val) + self.min_val
        _, sorted_indices = torch.sort(values)
        return values.unsqueeze(1), sorted_indices


class ConvexHullDataset(Dataset):
    def __init__(self, num_samples: int, set_size: int):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        points = torch.rand(self.set_size, 2)
        try:
            hull = ScipyConvexHull(points.numpy())
            hull_indices = hull.vertices
            if len(hull_indices) < self.set_size:
                padding = [hull_indices[0]] * (self.set_size - len(hull_indices))
                hull_indices = np.concatenate([hull_indices, padding])
            return points, torch.from_numpy(hull_indices).long()
        except Exception:
            return points, torch.argsort(points[:, 0])


class TSPDataset(Dataset):
    def __init__(self, num_samples: int, num_cities: int):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cities = torch.rand(self.num_cities, 2)
        return cities, self._greedy_tour(cities)
    
    def _greedy_tour(self, cities):
        n = len(cities)
        unvisited = set(range(n))
        current = 0
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return torch.tensor(tour, dtype=torch.long)


def train_epoch(model, dataloader, optimizer, device, use_teacher_forcing=True):
    model.train()
    total_loss, total_acc, num_batches = 0, 0, 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        lengths = torch.full((batch_size,), set_size, device=device)
        
        pointers, log_probs, _ = model(
            inputs, lengths, max_steps=set_size,
            teacher_forcing=targets if use_teacher_forcing else None
        )
        
        loss = compute_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += accuracy(pointers, targets)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_acc, num_batches = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            set_size = inputs.size(1)
            lengths = torch.full((inputs.size(0),), set_size, device=device)
            pointers, log_probs, _ = model(inputs, lengths, max_steps=set_size)
            total_loss += compute_loss(log_probs, targets).item()
            total_acc += accuracy(pointers, targets)
            num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


if __name__ == "__main__":
    print("Training script loaded.")
