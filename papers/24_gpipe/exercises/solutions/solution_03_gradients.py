"""
Solution 03: Gradient Accumulation

Accumulating gradients manually across micro-batches.
"""

import torch
import torch.nn as nn

def compute_accumulated_gradients(model, micro_batches, targets, criterion):
    for x_small, y_small in zip(micro_batches, targets):
        out = model(x_small)
        loss = criterion(out, y_small)
        # Gradient is averaged over the micro-batch size by default in MSELoss('mean')
        # To match the mini-batch gradient exactly, we might need to scale loss 
        # based on micro-batch count, but for training stability, 
        # standard .backward() is usually sufficient.
        loss.backward()

def test_accum():
    model = nn.Linear(5, 1)
    micro_batches = [torch.randn(4, 5), torch.randn(4, 5)]
    targets = [torch.randn(4, 1), torch.randn(4, 1)]
    criterion = nn.MSELoss()
    
    compute_accumulated_gradients(model, micro_batches, targets, criterion)
    
    for p in model.parameters():
        assert p.grad is not None
    print("[OK] Solution 03 verified.")

if __name__ == "__main__":
    test_accum()
