import torch
import numpy as np

def generate_nth_farthest_batch(batch_size, seq_len, vector_dim, n_index=0):
    """
    Generates a batch for the N-th Farthest task.
    
    Task: Given a sequence of vectors, output the vector that is farthest
    (Euclidean distance) from the N-th vector in the sequence.
    
    Inputs:
        batch_size: Number of samples.
        seq_len: Length of the sequence.
        vector_dim: Dimension of each vector.
        n_index: The index of the reference vector (default 0).
        
    Returns:
        inputs: (batch, seq_len, vector_dim) - Random vectors in [-1, 1]
        targets: (batch, vector_dim) - The vector that is farthest from the n-th vector
    """
    # 1. Generate random inputs in range [-1, 1]
    inputs = torch.rand(batch_size, seq_len, vector_dim) * 2 - 1
    targets = []
    
    for i in range(batch_size):
        seq = inputs[i]
        
        # TODO: Identify the reference vector at n_index
        ref_vec = # YOUR CODE HERE
        
        # TODO: Calculate Euclidean distance from ref_vec to all other vectors in seq
        # dists should be shape (seq_len,)
        dists = # YOUR CODE HERE
        
        # TODO: Find the index of the maximum distance
        farthest_idx = # YOUR CODE HERE
        
        # TODO: Get the target vector
        target_vec = # YOUR CODE HERE
        
        targets.append(target_vec)
        
    targets = torch.stack(targets)
    return inputs, targets

def test_generator():
    print("Testing Data Generator...")
    B, L, D = 2, 5, 2
    inputs, targets = generate_nth_farthest_batch(B, L, D)
    
    assert inputs.shape == (B, L, D)
    assert targets.shape == (B, D)
    
    # Check correctness manually for first sample
    seq = inputs[0]
    ref = seq[0]
    dists = torch.norm(seq - ref, dim=1)
    max_idx = torch.argmax(dists)
    expected = seq[max_idx]
    
    # Use allclose because of float precision
    assert torch.allclose(targets[0], expected), "Target verification failed!"
    print("Test Passed!")

if __name__ == "__main__":
    test_generator()
