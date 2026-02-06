"""
Solution 2: Convolutional Shift
===============================

Complete solution for Exercise 2.
"""

import torch

def circular_shift(w, s):
    """
    Args:
        w: Weighting vector [batch_size, N]
        s: Shift vector [batch_size, 3] (indices for shift -1, 0, 1)
        
    Returns:
        w_s: Shifted weighting [batch_size, N]
    """
    # Use torch.roll for efficient circular shift
    # s[:, 0] is for shift -1 (left), s[:, 1] is for shift 0, s[:, 2] is for shift 1 (right)
    # In torch.roll: positive shifts move to the right, negative to the left.
    
    # Initialize w_s
    w_s = torch.zeros_like(w)
    N = w.size(1)
    
    # shift -1 (moving right in indices means torch.roll with positive 1)
    # Graves paper uses s(i) meaning weight at i moves to i+1 if shift is +1.
    # We follow the convention in implementation.py:
    # s[0] -> shift -1
    # s[1] -> shift 0
    # s[2] -> shift +1
    
    w_s = (
        s[:, 0] * torch.roll(w, shifts=1, dims=1) +
        s[:, 1] * w +
        s[:, 2] * torch.roll(w, shifts=-1, dims=1)
    )
    
    return w_s

if __name__ == "__main__":
    # Test
    batch, N = 1, 10
    w = torch.zeros(batch, N)
    w[0, 5] = 1.0 
    
    # Shift +1 (index 2 in s is shift +1)
    s = torch.tensor([[0.0, 0.0, 1.0]])
    
    w_s = circular_shift(w, s)
    print("Shifted weighting:", w_s)
    assert w_s[0, 6] == 1.0
    print("Success")
