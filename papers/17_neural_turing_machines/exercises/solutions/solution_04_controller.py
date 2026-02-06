"""
Solution 4: Controller Input Preparation
========================================

Complete solution for Exercise 4.
"""

import torch

def prepare_controller_input(x, read_vectors):
    """
    Args:
        x: Current input [batch_size, input_size]
        read_vectors: List of Tensors, each [batch_size, memory_dim]
        
    Returns:
        full_input: Concatenated vector for the controller
    """
    # Simply concatenate along the feature dimension (dim=1)
    return torch.cat([x] + read_vectors, dim=1)

if __name__ == "__main__":
    # Test
    batch = 2
    x = torch.randn(batch, 5)
    r1 = torch.randn(batch, 8)
    r2 = torch.randn(batch, 8)
    
    full_inp = prepare_controller_input(x, [r1, r2])
    print("Full input shape:", full_inp.shape)
    assert full_inp.shape == (batch, 21)
    print("Success")
