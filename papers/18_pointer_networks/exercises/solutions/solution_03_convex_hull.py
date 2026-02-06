"""
Solution 3: Convex Hull Data Formatting
Standardized for Day 18 of 30u30

Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import numpy as np
from scipy.spatial import ConvexHull

def get_convex_hull_pointers(points):
    """
    Calculate the indices of points forming the convex hull.
    """
    # Simply use scipy's optimized ConvexHull which returns vertex indices in order.
    hull = ConvexHull(points)
    return hull.vertices.tolist()
