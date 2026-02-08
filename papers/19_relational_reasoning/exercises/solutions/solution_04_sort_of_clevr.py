"""
Solution 4: Sort-of-CLEVR Pair Generation
Standardized for Day 19 of 30u30
"""

import torch

def create_contextual_pairs(objects: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """
    Solution: Inject global question vector into local pairs.
    """
    batch_size, n_objects, object_dim = objects.shape
    
    # 1. Expand objects to create (o_i, o_j) grid
    # Shape: (B, N, N, D)
    o_i = objects.unsqueeze(2).expand(-1, -1, n_objects, -1)
    o_j = objects.unsqueeze(1).expand(-1, n_objects, -1, -1)
    
    # 2. Expand question to match the grid
    # Shape: (B, N, N, Q)
    q_expanded = question.unsqueeze(1).unsqueeze(2).expand(-1, n_objects, n_objects, -1)
    
    # 3. Concatenate everything
    # Output: (B, N, N, 2D + Q)
    return torch.cat([o_i, o_j, q_expanded], dim=-1)
