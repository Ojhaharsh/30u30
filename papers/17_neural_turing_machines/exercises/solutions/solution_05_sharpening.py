"""
Solution 5: Sharpening Mechanism
================================

Complete solution for Exercise 5.
"""

import torch

def sharpening(w, gamma):
    """
    Args:
        w: Blurry weighting vector [batch_size, N]
        gamma: Sharpening factor [batch_size, 1]
        
    Returns:
        w_sharp: Sharpened weighting [batch_size, N]
    """
    # 1. Raise w to the power of gamma
    # gamma: [batch, 1]
    # w: [batch, N]
    w_pow = w ** gamma
    
    # 2. Re-normalize
    w_sharp = w_pow / (w_pow.sum(dim=1, keepdim=True) + 1e-8)
    
    return w_sharp

if __name__ == "__main__":
    # Test
    batch, N = 1, 5
    w = torch.tensor([[0.2, 0.3, 0.2, 0.1, 0.2]])
    gamma = torch.tensor([[5.0]])
    
    w_sharp = sharpening(w, gamma)
    print("Original w:", w)
    print("Sharpened w:", w_sharp)
    assert w_sharp[0, 1] > w[0, 1]
    assert torch.allclose(w_sharp.sum(), torch.tensor(1.0))
    print("Success")
