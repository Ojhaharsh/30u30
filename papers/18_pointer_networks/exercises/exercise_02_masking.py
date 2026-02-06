"""
Exercise 2: Implement Attention Masking
Standardized for Day 18 of 30u30

Your task: Implement the masking logic to prevent the model from picking the same item twice.
Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch
import torch.nn.functional as F

def apply_mask(scores, mask):
    """
    TODO: Implement the masking logic.
    
    Args:
        scores (Tensor): Raw attention scores (batch, seq_len)
        mask (Tensor): Boolean mask (batch, seq_len) where True = indices to mask out
        
    Returns:
        log_probs (Tensor): Log-probabilities with masked positions effectively zeroed
    """
    # TODO: Use scores.masked_fill_ to replace masked positions with -inf
    # Then apply log_softmax
    pass

if __name__ == "__main__":
    print("Exercise 2: Attention Masking")
    print("=" * 50)
    
    # Test case
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[False, True, False, False]]) # Mask out index 1 (value 2.0)
    
    try:
        result = apply_mask(scores, mask)
        probs = torch.exp(result)
        
        print(f"Original scores: {scores.tolist()}")
        print(f"Mask: {mask.tolist()}")
        print(f"Masked probabilities: {probs.tolist()}")
        
        if probs[0, 1] < 1e-6:
            print("\n[OK] Masking successful! Index 1 was ignored.")
        else:
            print("\n[FAIL] Index 1 was NOT ignored. Check your masked_fill logic.")
            
    except Exception as e:
        print(f"\n[FAIL] Error occurred: {e}")
