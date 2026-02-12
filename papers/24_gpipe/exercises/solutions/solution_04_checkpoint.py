"""
Solution 04: Re-materialization (Checkpointing)

Using torch.utils.checkpoint.checkpoint.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def checkpointed_forward(layer_seq, x):
    # use_reentrant=False is the modern default for checkpointing
    return checkpoint(layer_seq, x, use_reentrant=False)

def test_checkpoint():
    layers = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    x = torch.randn(1, 10, requires_grad=True)
    
    out = checkpointed_forward(layers, x)
    assert out.shape == (1, 10)
    
    out.sum().backward()
    assert x.grad is not None
    print("[OK] Solution 04 verified.")

if __name__ == "__main__":
    test_checkpoint()
