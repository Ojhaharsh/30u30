"""
Exercise 1: Implement the Pointer Attention mechanism
Standardized for Day 18 of 30u30

Your task: Complete the PointerAttention module to calculate additive attention scores.
Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerAttention(nn.Module):
    """
    Pointer Attention Mechanism.
    
    TODO: Fill in the __init__ and forward methods below.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # TODO: Initialize three linear layers (W1, W2, v)
        # Hint: W1 and W2 map hidden_size -> hidden_size. v maps hidden_size -> 1.
        # Use bias=False as per typical implementation.
        self.W1 = None # TODO
        self.W2 = None # TODO
        self.v = None  # TODO

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Forward pass for attention scores.
        
        Args:
            decoder_hidden: (batch, hidden_size)
            encoder_outputs: (batch, seq_len, hidden_size)
            mask: (batch, seq_len)
            
        Returns:
            log_probs: (batch, seq_len)
        """
        # TODO: Implement the Pointer Attention formula (Eq. 3)
        # 1. Project decoder_hidden and encoder_outputs using W2 and W1
        # 2. Combine them (sum) and apply tanh
        # 3. Project the result using v and squeeze the last dimension
        # 4. If mask is provided, use masked_fill with float('-inf')
        # 5. Return log_softmax over the last dimension

        pass # TODO

if __name__ == "__main__":
    print("Exercise 1: Pointer Attention")
    print("=" * 50)
    print("\nYour task: Fill in the TODOs in exercise_01_pointer_attention.py")
    
    # Test your implementation
    hidden_size = 64
    batch_size = 2
    seq_len = 5
    
    try:
        attn = PointerAttention(hidden_size)
        d = torch.randn(batch_size, hidden_size)
        e = torch.randn(batch_size, seq_len, hidden_size)
        out = attn(d, e)
        
        print(f"\n[OK] Output shape: {out.shape}")
        print("Now check if your logic matches Equation 3 in the paper!")
    except Exception as e:
        print(f"\n[FAIL] Error occurred: {e}")
