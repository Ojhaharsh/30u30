"""
Exercise 02: Pipeline Forward Pass

Once micro-batches are created, they must pass through a sequence 
of model partitions (stages). In this exercise, you implement the 
simplified sequential forward pass for a pipeline.
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
    """
    Simulate a synchronous pipeline forward pass.
    
    Args:
        stages (List[nn.Module]): List of K sequential stages.
        micro_batches (List[torch.Tensor]): List of M micro-batches.
        
    Returns:
        List[torch.Tensor]: Outputs of the final stage for each micro-batch.
        
    TODO: Implement the nested loop or logic that moves each 
    micro-batch through every stage.
    """
    # YOUR CODE HERE
    raise NotImplementedError()

def test_pipeline():
    # K=2 stages, M=2 micro-batches
    stages = [SimpleStage(nn.Linear(10, 10)), SimpleStage(nn.Linear(10, 5))]
    micro_batches = [torch.randn(4, 10), torch.randn(4, 10)]
    
    outputs = pipeline_forward(stages, micro_batches)
    
    assert len(outputs) == 2
    assert outputs[0].shape == (4, 5)
    print("[OK] Exercise 02 passed.")

if __name__ == "__main__":
    test_pipeline()
