"""
Solution 4: TSP Tour Cost Calculation
Standardized for Day 18 of 30u30

Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch
import numpy as np

def calculate_tour_cost(points, pointers):
    """
    Calculate the total length of the TSP tour.
    """
    cost = 0.0
    num_cities = len(pointers)
    
    for i in range(num_cities):
        # Current city index
        c1 = pointers[i]
        # Next city index (wrap around to 0 at the end)
        c2 = pointers[(i + 1) % num_cities]
        
        # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
        p1, p2 = points[c1], points[c2]
        dist = torch.sqrt(torch.sum((p2 - p1)**2))
        cost += dist.item()
        
    return cost
