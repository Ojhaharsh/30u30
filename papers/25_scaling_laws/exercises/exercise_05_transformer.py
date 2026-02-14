"""
Day 25 Exercise 5: Minimal Scaling Transformer

Your final task is to implement a Transformer block and calculate its
non-embedding parameters manually to verify the formula:
N approx 12 * L * d_model^2

Instructions:
1. Implement the ScalingBlock (Attention + MLP).
2. Calculate the parameter count N for the block.
3. Verify if it matches the 12 * L * d_model^2 rule of thumb.
"""

import torch
import torch.nn as nn

class ScalingBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # TODO: Implement a standard Transformer block
        # 1. Multi-head Attention (Linear layers for Q, K, V, O)
        # 2. MLP (Linear layer 4*d_model, then Linear back to d_model)
        # 3. LayerNorms
        pass

    def count_params(self):
        """
        TODO: Return the total number of trainable parameters in this block.
        """
        # YOUR CODE HERE
        pass

if __name__ == "__main__":
    d_model = 128
    block = ScalingBlock(d_model)
    
    if hasattr(block, 'count_params') and block.count_params() is not None:
        actual_n = block.count_params()
        theoretical_n = 12 * (d_model**2)
        
        print(f"D_Model: {d_model}")
        print(f"Actual Params: {actual_n}")
        print(f"Theory (12*d^2): {theoretical_n}")
        
        diff = abs(actual_n - theoretical_n) / theoretical_n
        if diff < 0.1:
            print("[OK] Parameter count matches the Transformer scaling rule.")
        else:
            print("[NOTE] Count differs from rule of thumb. Did you include LayerNorms or Biases?")
    else:
        print("[FAIL] ScalingBlock not implemented.")
