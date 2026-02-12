"""
Exercise 04: Re-materialization (Checkpointing)

Section 3.2 of the GPipe paper describes re-materialization. 
In PyTorch, this is achieved using torch.utils.checkpoint.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def checkpointed_forward(layer_seq, x):
    """
    Perform a forward pass through layer_seq using checkpointing.
    
    Args:
        layer_seq (nn.Sequential): Sequence of layers.
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Output tensor.
        
    TODO: Wrap the forward pass in torch.utils.checkpoint.checkpoint.
    Note: PyTorch requires inputs to have 'requires_grad=True' for 
    checkpointing to work correctly in some versions.
    """
    # YOUR CODE HERE
    raise NotImplementedError()

def test_checkpoint():
    layers = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    x = torch.randn(1, 10, requires_grad=True)
    
    out = checkpointed_forward(layers, x)
    assert out.shape == (1, 10)
    
    # Verify backward pass works
    out.sum().backward()
    assert x.grad is not None
    print("[OK] Exercise 04 passed.")

if __name__ == "__main__":
    test_checkpoint()
