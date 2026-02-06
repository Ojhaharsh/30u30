"""
Exercise 3: Convex Hull Data Formatting
Standardized for Day 18 of 30u30

Your task: Implement a function to calculate the "pointer" targets for a Convex Hull task.
Given a set of points, you need to find the indices of the points that form the convex hull
in counter-clockwise order.

This is a classic combinatorial task from the Ptr-Net paper (Section 3.1, results in Section 4.2).
"""

import numpy as np
import torch

def get_convex_hull_pointers(points):
    """
    Calculate the indices of points forming the convex hull.
    
    Args:
        points (ndarray): (N, 2) array of (x, y) coordinates.
        
    Returns:
        indices (list of int): Indices of input points in CCW hull order.
    """
    # TODO: Implement a simple Convex Hull algorithm (like Graham Scan or Monotone Chain)
    # or use a library like scipy.spatial.ConvexHull.
    # For this exercise, assume you have N points and return the indices.
    
    # Hint: scipy.spatial.ConvexHull(points).vertices returns the indices in order.
    pass

if __name__ == "__main__":
    print("Exercise 3: Convex Hull Formatting")
    print("=" * 50)
    
    # Test points (a square with one point inside)
    points = np.array([
        [0, 0],   # 0
        [1, 0],   # 1
        [1, 1],   # 2
        [0, 1],   # 3
        [0.5, 0.5] # 4 (Inside)
    ])
    
    try:
        from scipy.spatial import ConvexHull
        # Logic check:
        hull_indices = get_convex_hull_pointers(points)
        
        print(f"Input Points:\n{points}")
        print(f"Your Hull Indices: {hull_indices}")
        
        # Validation
        expected = ConvexHull(points).vertices.tolist()
        # CCW order might start at different points, so check if set and sequence are correct
        if set(hull_indices) == set(expected) and len(hull_indices) == len(expected):
            print("\n[OK] Correct points selected for the hull!")
        else:
            print(f"\n[FAIL] Expected indices like {expected}, but got {hull_indices}")
            
    except ImportError:
        print("\n[SKIP] Scipy not found. Please install it to run the validation.")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
