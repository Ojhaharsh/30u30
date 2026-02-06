"""
Exercise 4: TSP Tour Cost Calculation
Standardized for Day 18 of 30u30

Your task: Implement a function to calculate the total Euclidean distance of a TSP tour
given a sequence of pointers (indices).

This will help you understand:
- How pointers relate to the original coordinate space.
- Why minimizing tour length is the objective for Ptr-Nets in TSP.
"""

import torch
import numpy as np

def calculate_tour_cost(points, pointers):
    """
    Calculate the total length of the TSP tour.
    
    Args:
        points (Tensor): (N, 2) coordinates of the cities.
        pointers (list of int): Sequence of city indices (the tour).
        
    Returns:
        cost (float): Total Euclidean distance.
    """
    # TODO:
    # 1. Iterate through the pointers.
    # 2. Sum the distances between consecutive cities: points[p1] to points[p2].
    # 3. Don't forget to close the loop: last city back to the first.
    
    pass

if __name__ == "__main__":
    print("Exercise 4: TSP Cost Analysis")
    print("=" * 50)
    
    # Test cities (A 1x1 unit square)
    cities = torch.tensor([
        [0.0, 0.0], # 0
        [1.0, 0.0], # 1
        [1.0, 1.0], # 2
        [0.0, 1.0]  # 3
    ])
    
    # Tour: 0 -> 1 -> 2 -> 3 -> 0
    pointers = [0, 1, 2, 3]
    
    try:
        cost = calculate_tour_cost(cities, pointers)
        print(f"Cities:\n{cities}")
        print(f"Tour: {pointers}")
        print(f"Your calculated cost: {cost:.4f}")
        
        # Validation: Square perimeter = 1+1+1+1 = 4.0
        if abs(cost - 4.0) < 1e-4:
            print("\n[OK] Cost calculation is correct!")
        else:
            print(f"\n[FAIL] Expected cost 4.0, but got {cost}")
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
