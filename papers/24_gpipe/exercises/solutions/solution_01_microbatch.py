"""
Solution 01: Micro-batch Splitting

Reference implementation using torch.chunk.
"""

import torch

def split_mini_batch(x, n_microbatches):
    # Using torch.chunk for optimized slicing
    return torch.chunk(x, n_microbatches, dim=0)

def test_split():
    N, C, H, W = 16, 3, 224, 224
    M = 4
    x = torch.randn(N, C, H, W)
    
    micro_batches = split_mini_batch(x, M)
    
    assert len(micro_batches) == M
    assert micro_batches[0].shape[0] == N // M
    assert torch.allclose(torch.cat(micro_batches, dim=0), x)
    print("[OK] Solution 01 verified.")

if __name__ == "__main__":
    test_split()
