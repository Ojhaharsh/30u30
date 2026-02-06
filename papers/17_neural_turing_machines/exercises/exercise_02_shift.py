"""
Exercise 2: Convolutional Shift
===============================

NTMs implementations location-based addressing via circular convolution 
(Graves et al., 2014, Section 3.3.1, Equation 8).

Formula:
w_s(i) = Σ_{j=0}^{N-1} w_g(j) s(i - j)

For a shift range of 3 (indices 0, 1, 2 representing shifts -1, 0, 1), 
Equation 8 simplifies to:
w_s(i) = s(0)*w_g(i+1) + s(1)*w_g(i) + s(2)*w_g(i-1)
"""

import torch

def circular_shift(w, s):
    """
    Args:
        w: Gated weighting vector [batch_size, N]
        s: Shift weighting [batch_size, 3] (for shifts -1, 0, +1)
        
    Returns:
        w_s: Shifted weighting [batch_size, N]
    """
    # TODO: Implement circular shift using torch.roll (Eq 8)
    pass

if __name__ == "__main__":
    # Test your implementation
    batch, N = 1, 10
    w = torch.zeros(batch, N)
    w[0, 5] = 1.0 # Focus on slot 5
    
    # Shift +1 (index 2 in s)
    s = torch.tensor([[0.0, 0.0, 1.0]])
    
    w_s = circular_shift(w, s)
    
    if w_s is not None:
        print(f"Post-shift focus index: {w_s[0].argmax().item()}")
        assert w_s[0, 6] == 1.0, "Focus should have shifted exactly to index 6"
