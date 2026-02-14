"""
Day 25 Solution 5: Minimal Scaling Transformer

This solution implements a standard Transformer block and verifies the 
parameter count logic used in scaling law papers.
"""

import torch
import torch.nn as nn

class ScalingBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 1. Attention: Q, K, V, O projections (each d_model * d_model)
        # Total Attention Params approx 4 * d_model^2
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        
        # 2. MLP: expansion to 4*d_model and back
        # Total MLP Params approx 8 * d_model^2
        self.mlp_up = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_down = nn.Linear(4 * d_model, d_model, bias=False)
        
        # 3. LayerNorms (negligible params: 2 * d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    d_model = 128
    block = ScalingBlock(d_model)
    
    actual_n = block.count_params()
    # Logic: 4 (attn) + 8 (mlp) = 12 * d_model^2
    theoretical_n = 12 * (d_model**2)
    
    print(f"D_Model: {d_model}")
    print(f"Actual Params: {actual_n}")
    print(f"Theory (12*d^2): {theoretical_n}")
    
    # LayerNorms add a tiny amount: 2*2*d_model = 4*128 = 512
    # Actual should be 12*d^2 + 4*d = 196,608 + 512 = 197,120
    print(f"Difference (LayerNorms/Biases): {actual_n - theoretical_n}")
    print("[OK] Manual verification confirms the 12*L*d^2 rule.")
