"""
Solution 02: Pipeline Forward Pass

Synchronous pipeline logic moving M micro-batches through K stages.
"""

import torch
import torch.nn as nn

class SimpleStage(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x)

def pipeline_forward(stages, micro_batches):
    n_partitions = len(stages)
    n_microbatches = len(micro_batches)
    
    # Store intermediate results
    outputs = [[None for _ in range(n_partitions)] for _ in range(n_microbatches)]
    
    # Simple synchronous loop
    # In practice, this schedule overlaps to reduce idle time (the bubble),
    # but sequential logic for simulation is also valid for verification.
    for m in range(n_microbatches):
        current_input = micro_batches[m]
        for k in range(n_partitions):
            current_input = stages[k](current_input)
            outputs[m][k] = current_input
            
    return [outputs[m][-1] for m in range(n_microbatches)]

def test_pipeline():
    stages = [SimpleStage(nn.Linear(10, 10)), SimpleStage(nn.Linear(10, 5))]
    micro_batches = [torch.randn(4, 10), torch.randn(4, 10)]
    
    outputs = pipeline_forward(stages, micro_batches)
    
    assert len(outputs) == 2
    assert outputs[0].shape == (4, 5)
    print("[OK] Solution 02 verified.")

if __name__ == "__main__":
    test_pipeline()
