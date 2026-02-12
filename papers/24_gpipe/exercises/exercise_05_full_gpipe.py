"""
Exercise 05: Full GPipe Integration

The final challenge: implement the GPipe wrapper that handles 
partitioning, micro-batching, and coordinated forwarding.
"""

import torch
import torch.nn as nn

class GPipeExercise(nn.Module):
    def __init__(self, model, n_partitions, n_microbatches):
        super().__init__()
        self.n_partitions = n_partitions
        self.n_microbatches = n_microbatches
        
        # TODO: Partition the sequential model into K stages
        self.stages = nn.ModuleList()
        # YOUR CODE HERE
        
    def forward(self, x):
        """
        TODO: 
        1. Split x into M micro-batches.
        2. Pass each through the stages.
        3. Concatenate and return.
        """
        # YOUR CODE HERE
        raise NotImplementedError()

def test_full():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
    gpipe = GPipeExercise(model, n_partitions=2, n_microbatches=4)
    
    x = torch.randn(16, 10)
    out = gpipe(x)
    
    assert out.shape == (16, 5)
    print("[OK] Exercise 05 passed.")

if __name__ == "__main__":
    test_full()
