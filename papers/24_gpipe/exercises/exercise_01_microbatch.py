"""
Exercise 01: Micro-batch Splitting

GPipe's core efficiency comes from splitting mini-batches into 
smaller micro-batches (Section 3.1). In this exercise, you will 
implement the logic to chunk a tensor while handling edge cases.
"""

import torch

def split_mini_batch(x, n_microbatches):
    """
    Split a mini-batch x into M micro-batches.
    
    Args:
        x (torch.Tensor): Input mini-batch tensor (N, C, ...)
        n_microbatches (int): Number of micro-batches (M)
        
    Returns:
        List[torch.Tensor]: List of micro-batch tensors.
        
    TODO: Implement batch splitting using torch.chunk or similar.
    Ensure that the operation is performant and maintains batch-size consistency.
    """
    # YOUR CODE HERE
    raise NotImplementedError()

def test_split():
    N, C, H, W = 16, 3, 224, 224
    M = 4
    x = torch.randn(N, C, H, W)
    
    micro_batches = split_mini_batch(x, M)
    
    assert len(micro_batches) == M, f"Expected {M} chunks, got {len(micro_batches)}"
    assert micro_batches[0].shape[0] == N // M, "Micro-batch size mismatch"
    assert torch.allclose(torch.cat(micro_batches, dim=0), x), "Reconstruction failed"
    print("[OK] Exercise 01 passed.")

if __name__ == "__main__":
    test_split()
