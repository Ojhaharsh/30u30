"""
Solution 3: Memory Update (Erase and Add)
=========================================

Complete solution for Exercise 3.
"""

import torch

def memory_update(w, erase, add, prev_memory):
    """
    Args:
        w: Write weighting [batch_size, N]
        erase: Erase vector [batch_size, M]
        add: Add vector [batch_size, M]
        prev_memory: Memory matrix [batch_size, N, M]
        
    Returns:
        new_memory: Updated memory matrix [batch_size, N, M]
    """
    # 1. Compute erase term
    # w is [batch, N], erase is [batch, M]. Outer product -> [batch, N, M]
    erase_term = torch.matmul(w.unsqueeze(2), erase.unsqueeze(1))
    
    # 2. Apply erase to prev_memory
    memory_after_erase = prev_memory * (1 - erase_term)
    
    # 3. Compute add term
    add_term = torch.matmul(w.unsqueeze(2), add.unsqueeze(1))
    
    # 4. Apply add to memory
    new_memory = memory_after_erase + add_term
    
    return new_memory

if __name__ == "__main__":
    # Test
    batch, N, M = 1, 5, 4
    prev_memory = torch.ones(batch, N, M)
    w = torch.zeros(batch, N)
    w[0, 2] = 1.0 
    
    erase = torch.ones(batch, M) 
    add = torch.tensor([[1.2, 3.4, 5.6, 7.8]])
    
    new_memory = memory_update(w, erase, add, prev_memory)
    print("New memory at slot 2:", new_memory[0, 2])
    assert torch.allclose(new_memory[0, 2], add[0])
    print("Success")
