"""
Exercise 3: Memory Update (Erase and Add)
=========================================

The NTM write head update is defined as a sequence of Erase and Add operations 
(Graves et al., 2014, Section 3.3, Equations 3 and 4).

Formulas:
Erase (Eq 3): M_t_erase(i) = M_{t-1}(i) * [1 - w_t(i) * e_t]
Add   (Eq 4): M_t(i) = M_t_erase(i) + w_t(i) * a_t
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
    # TODO: Implement Eq 3 and Eq 4
    # 1. Compute erase term (outer product of w and erase)
    # 2. Compute add term (outer product of w and add)
    pass

if __name__ == "__main__":
    # Test your implementation
    batch, N, M = 1, 5, 4
    prev_memory = torch.ones(batch, N, M)
    w = torch.zeros(batch, N)
    w[0, 2] = 1.0 # Target slot 2
    
    erase = torch.ones(batch, M) # Total erase
    add = torch.tensor([[1.2, 3.4, 5.6, 7.8]]) # Selective add
    
    new_memory = memory_update(w, erase, add, prev_memory)
    
    if new_memory is not None:
        print(f"Slot 2 content: {new_memory[0, 2].tolist()}")
        assert torch.allclose(new_memory[0, 2], add[0])
        assert torch.allclose(new_memory[0, 0], prev_memory[0, 0])
