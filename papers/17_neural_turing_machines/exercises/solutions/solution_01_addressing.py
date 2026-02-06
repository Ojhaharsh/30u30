"""
Solution 1: Addressing Mechanism - Content Addressing
=====================================================

Implements the content addressing logic from Graves et al. (2014), Section 3.3.1.
This mechanism allows the controller to retrieve information based on 
similarity to its current state.
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
    # 1. Cosine similarity K[k, M] (Equation 6)
    # We normalize both k and memory vectors to unit length
    norm_k = k.norm(p=2, dim=1, keepdim=True) + 1e-8
    norm_mem = memory.norm(p=2, dim=2) + 1e-8
    
    # Batch matrix multiplication to get dot products across all memory slots
    # [batch, N, M] @ [batch, M, 1] -> [batch, N, 1]
    dot_product = torch.matmul(memory, k.unsqueeze(2)).squeeze(2)
    sim = dot_product / (norm_k * norm_mem)
    
    # 2. Key Strength & Softmax (Equation 5)
    # Beta controls the precision of the focus. High beta = sharp focus.
    w_c = F.softmax(beta * sim, dim=1)
    
    return w_c

if __name__ == "__main__":
    # Regression test
    batch, N, M = 1, 10, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    beta = torch.tensor([[10.0]])
    
    memory = torch.randn(batch, N, M)
    memory[0, 0] = k[0]
    
    w = content_addressing(k, beta, memory)
    print(f"Content focus at slot 0: {w[0, 0].item():.4f}")
    assert w[0, 0] > 0.9
