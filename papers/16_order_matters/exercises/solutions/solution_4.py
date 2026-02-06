"""
Solution 4: Convex Hull with Pointer Networks
==============================================

Implementation of a geometry-aware Pointer Network.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial import ConvexHull as ScipyConvexHull
import sys
sys.path.append('..')
from implementation import ReadProcessWrite


class ConvexHullDataset(Dataset):
    def __init__(self, num_samples, num_points):
        self.num_samples = num_samples
        self.num_points = num_points
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        points = torch.rand(self.num_points, 2)
        try:
            hull = ScipyConvexHull(points.numpy())
            hull_indices = hull.vertices
            if len(hull_indices) < self.num_points:
                padding = [hull_indices[0]] * (self.num_points - len(hull_indices))
                hull_indices = np.concatenate([hull_indices, padding])
            return points, torch.from_numpy(hull_indices).long()
        except:
            return points, torch.argsort(points[:, 0])


def train_convex_hull():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReadProcessWrite(input_dim=2, hidden_dim=128, use_set_encoder=True).to(device)
    dataset = ConvexHullDataset(1000, 10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for points, targets in loader:
        points, targets = points.to(device), targets.to(device)
        lengths = torch.full((points.size(0),), 10, device=device)
        _, log_probs, _ = model(points, lengths, 10, teacher_forcing=targets)
        
        loss = -log_probs.gather(1, targets).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    print("Convex Hull solution loaded.")
