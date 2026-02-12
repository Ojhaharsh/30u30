"""
Solution 05: Full GPipe Integration

Complete wrapper combining all concepts.
"""

import torch
import torch.nn as nn

class GPipeExercise(nn.Module):
    def __init__(self, model, n_partitions, n_microbatches):
        super().__init__()
        self.n_partitions = n_partitions
        self.n_microbatches = n_microbatches
        
        # Correctly partitioning the model
        total_layers = len(model)
        chunk_size = total_layers // n_partitions
        
        self.stages = nn.ModuleList()
        for i in range(n_partitions):
            start = i * chunk_size
            end = (i+1) * chunk_size if i < n_partitions - 1 else total_layers
            self.stages.append(model[start:end])
        
    def forward(self, x):
        # 1. Split
        micro_batches = torch.chunk(x, self.n_microbatches, dim=0)
        
        # 2. Sequential forward through stages for each micro-batch
        final_outputs = []
        for m_batch in micro_batches:
            current = m_batch
            for stage in self.stages:
                current = stage(current)
            final_outputs.append(current)
            
        # 3. Join
        return torch.cat(final_outputs, dim=0)

def test_full():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
    gpipe = GPipeExercise(model, n_partitions=2, n_microbatches=4)
    
    x = torch.randn(16, 10)
    out = gpipe(x)
    
    assert out.shape == (16, 5)
    print("[OK] Solution 05 verified.")

if __name__ == "__main__":
    test_full()
