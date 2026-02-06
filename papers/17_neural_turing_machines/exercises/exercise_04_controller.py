"""
Exercise 4: Controller Input Preparation
========================================

Decoupling processing from memory requires the NTM controller to observe 
the state of memory at the previous timestep (Graves et al., 2014, Section 2).

Task:
Implement the concatenation of input vector x and read vectors r_{t-1}. 
This combined vector forms the input to the RNN/Controller.
"""

import torch

def prepare_controller_input(x, read_vectors):
    """
    Args:
        x: Current input [batch_size, input_size]
        read_vectors: List of read head vectors, each [batch_size, M]
        
    Returns:
        full_input: Concatenated vector [batch_size, input_size + num_heads * M]
    """
    # TODO: Concatenate x and all read vectors along feature dimension
    pass

if __name__ == "__main__":
    # Test
    batch = 2
    x = torch.randn(batch, 5)
    r1 = torch.randn(batch, 8)
    r2 = torch.randn(batch, 8)
    
    full_inp = prepare_controller_input(x, [r1, r2])
    if full_inp is not None:
        print(f"Resulting dimension: {full_inp.shape[1]}")
        assert full_inp.shape[1] == 21
