"""
Exercise 03: Gradient Accumulation

In GPipe, gradients are accumulated across all micro-batches before 
the optimizer update (synchronous update). In this exercise, you 
will implement the backward pass logic for multiple micro-batches.
"""

import torch
import torch.nn as nn

def compute_accumulated_gradients(model, micro_batches, targets, criterion):
    """
    Perform forward and backward passes for all micro-batches and 
    accumulate gradients in the model parameters.
    
    Args:
        model (nn.Module): The partition/stage.
        micro_batches (List[torch.Tensor]): Data split into M chunks.
        targets (List[torch.Tensor]): Correct labels split into M chunks.
        criterion (nn.Module): Loss function.
        
    TODO: 
    1. Iterate through micro-batches.
    2. Compute forward pass.
    3. Compute loss.
    4. Call .backward().
    Note: Gradients accumulate automatically in PyTorch if not zeroed.
    """
    # YOUR CODE HERE
    raise NotImplementedError()

def test_accum():
    model = nn.Linear(5, 1)
    # 2 micro-batches of size 4
    micro_batches = [torch.randn(4, 5), torch.randn(4, 5)]
    targets = [torch.randn(4, 1), torch.randn(4, 1)]
    criterion = nn.MSELoss()
    
    compute_accumulated_gradients(model, micro_batches, targets, criterion)
    
    # Check if gradients are populated
    for p in model.parameters():
        assert p.grad is not None, "Gradients not accumulated"
    print("[OK] Exercise 03 passed.")

if __name__ == "__main__":
    test_accum()
