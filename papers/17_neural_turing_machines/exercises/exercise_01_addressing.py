"""
Exercise 1: Addressing Mechanism - Content Addressing
=====================================================

Implementing the first step of the NTM addressing pipeline as defined in 
Graves et al. (2014), Section 3.3.1, Equation 5.

The goal is to produce a weighting vector 'w' where each element represents 
how much the head should focus on a specific memory location based on 
its similarity to a search key 'k'.

Formula (Eq 5 & 6):
w_c(i) = exp(beta * K[k, M(i)]) / Σ exp(beta * K[k, M(j)])
Where K[., .] is the cosine similarity.
"""

import torch
import torch.nn.functional as F

def content_addressing(k, beta, memory):
    """
    Args:
        k: Search key vector [batch_size, M]
        beta: Key strength [batch_size, 1]
        memory: Memory matrix [batch_size, N, M]
        
    Returns:
        w_c: Content-based weighting [batch_size, N]
    """
    # TODO: Implement content-based addressing
    # 1. Calculate cosine similarity K[k, M]
    # 2. Multiply by beta factor
    # 3. Apply softmax over memory locations (dim=1)
    pass

if __name__ == "__main__":
    # Test your implementation
    batch, N, M = 1, 10, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    beta = torch.tensor([[10.0]]) # Sharp strength
    
    memory = torch.randn(batch, N, M)
    memory[0, 0] = k[0] # Exact match at slot 0
    
    w = content_addressing(k, beta, memory)
    
    if w is not None:
        print(f"Weight at slot 0: {w[0, 0].item():.4f}")
        assert w[0, 0] > 0.9, "Weighting should focus on the matching slot"
