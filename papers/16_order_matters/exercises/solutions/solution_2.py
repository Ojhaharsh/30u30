"""
Solution 2: Order-Invariant Set Encoder
========================================

Implementation of a Set Encoder using permutation-invariant self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """
    Self-attention based encoder for sets.
    Omitting positional encodings ensures the output is invariant to 
    input permutations.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, set_size, input_dim]
        """
        # Step 1: Embedding
        x = self.embedding(x)
        
        # Step 2: Self-Attention (Order-Invariant)
        # Note: No positional embeddings are added here.
        attn_out, _ = self.self_attention(x, x, x)
        
        # Step 3: Normalize + Residual
        x = self.norm1(x + attn_out)
        
        # Step 4: Feedforward
        ff_out = self.ff(x)
        
        # Step 5: Normalize + Residual
        x = self.norm2(x + ff_out)
        
        return x


def test_order_invariance():
    print("Validating Order-Invariant Encoder")
    print("-" * 30)
    
    set_size = 5
    input_dim = 2
    hidden_dim = 32
    
    encoder = SetEncoder(input_dim, hidden_dim)
    encoder.eval()
    
    points = torch.rand(1, set_size, input_dim)
    indices = torch.randperm(set_size)
    points_permuted = points[:, indices]
    
    with torch.no_grad():
        encoded1 = encoder(points)
        encoded2 = encoder(points_permuted)
    
    agg1 = encoded1.mean(dim=1)
    agg2 = encoded2.mean(dim=1)
    
    assert torch.allclose(agg1, agg2, atol=1e-5)
    print("Permutation invariance verified via mean pooling.")
    print("All validation tests complete")

if __name__ == "__main__":
    test_order_invariance()
