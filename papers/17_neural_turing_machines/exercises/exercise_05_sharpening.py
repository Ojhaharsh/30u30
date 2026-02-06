"""
Exercise 5: Sharpening Mechanism
================================

Weightings in NTM addressing often become 'blurry' due to the circular 
convolution (Eq 8). Sharpening refines the weighting to prevent focus leakage.

Formula (Graves et al., 2014, Section 3.3.1, Eq 9):
w_sharp[i] = w[i]^gamma / Σ w[j]^gamma

Note:
- gamma (γ) should be ≥ 1.
- As γ increases, the weighting becomes more "peaked" or "sharp".
"""

import torch

def sharpening(w, gamma):
    """
    Args:
        w: Weighting vector [batch_size, N]
        gamma: Sharpening factor [batch_size, 1]
        
    Returns:
        w_sharp: Sharpened weighting [batch_size, N]
    """
    # TODO: Implement Eq 9
    # 1. Raise w to the power of gamma
    # 2. Re-normalize along dim 1
    pass

if __name__ == "__main__":
    # Test your implementation
    batch, N = 1, 5
    w = torch.tensor([[0.2, 0.3, 0.2, 0.1, 0.2]])
    gamma = torch.tensor([[5.0]])
    
    w_sharp = sharpening(w, gamma)
    
    if w_sharp is not None:
        print(f"Max weight before: {w.max().item():.2f}")
        print(f"Max weight after:  {w_sharp.max().item():.2f}")
        assert w_sharp[0, 1] > w[0, 1], "Sharpening should increase the peak value"
        assert torch.allclose(w_sharp.sum(), torch.tensor(1.0)), "Weighting must sum to 1.0"
