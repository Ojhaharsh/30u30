import torch

def generate_pairs_via_broadcasting(objects: torch.Tensor) -> torch.Tensor:
    """
    Solution 1: Pairwise object concatenation using broadcasting.
    """
    batch_size, num_objects, object_dim = objects.size()
    
    # Expand objects for Cartesian product
    o_i = objects.unsqueeze(2).repeat(1, 1, num_objects, 1)
    o_j = objects.unsqueeze(1).repeat(1, num_objects, 1, 1)
    
    # Concatenate pairs
    pairs = torch.cat([o_i, o_j], dim=-1)
    
    # Flatten N x N grid into N^2 sequence
    return pairs.view(batch_size, num_objects * num_objects, -1)
