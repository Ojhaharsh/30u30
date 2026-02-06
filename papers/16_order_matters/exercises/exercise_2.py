"""
Exercise 2: Order-Invariant Set Encoder
========================================

Learn how to encode sets where input order is arbitrary but semantics 
remain fixed.

Concept: An order-invariant encoder should satisfy f(permute(X)) = f(X) 
for any permutation of inputs.

Requirement: Build a set encoder using self-attention without positional vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """
    Order-invariant encoder using self-attention.
    Self-attention is natively permutation-invariant if positional 
    encodings are omitted.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # TODO: Embedding layer to project inputs to hidden_dim
        self.embedding = None 
        
        # TODO: Multi-head self-attention
        # CRITICAL: Do NOT add positional encodings.
        self.self_attention = None 
        
        # TODO: Feedforward network (hidden_dim -> hidden_dim*4 -> hidden_dim)
        self.ff = None 
        
        # TODO: Layer normalization
        self.norm1 = None
        self.norm2 = None
        
    def forward(self, x):
        """
        Encode a set in an order-invariant way.
        
        Args:
            x: [batch, set_size, input_dim]
            
        Returns:
            encoded: [batch, set_size, hidden_dim]
        """
        # TODO: Step 1 - Embed inputs
        x = None 
        
        # TODO: Step 2 - Self-attention
        attn_out, _ = None, None 
        
        # TODO: Step 3 - Residual connection + layer norm
        x = None 
        
        # TODO: Step 4 - Feedforward
        ff_out = None 
        
        # TODO: Step 5 - Residual connection + layer norm
        x = None 
        
        return x


def test_order_invariance():
    """Verify the encoder satisfies permutation invariance."""
    print("Testing Order Invariance")
    print("-" * 30)
    
    batch_size = 1
    set_size = 5
    input_dim = 2
    hidden_dim = 32
    
    encoder = SetEncoder(input_dim, hidden_dim, num_heads=4)
    encoder.eval()
    
    # Create original set
    points = torch.rand(batch_size, set_size, input_dim)
    
    # Create permuted set
    indices = torch.randperm(set_size)
    points_permuted = points[:, indices]
    
    with torch.no_grad():
        encoded1 = encoder(points)
        encoded2 = encoder(points_permuted)
    
    # Check if the set of output vectors is the same (ignoring order)
    e1_sorted = torch.sort(encoded1[0].sum(dim=1))[0]
    e2_sorted = torch.sort(encoded2[0].sum(dim=1))[0]
    
    diff = (e1_sorted - e2_sorted).abs().mean()
    print(f"Mean difference after permutation: {diff.item():.6f}")
    
    # Aggregated representation (mean pooling) should be identical
    agg1 = encoded1.mean(dim=1)
    agg2 = encoded2.mean(dim=1)
    diff_agg = (agg1 - agg2).abs().mean()
    print(f"Difference in mean-pooled representation: {diff_agg.item():.6f}")

    print("-" * 30)

if __name__ == "__main__":
    test_order_invariance()
    
    print("Exercise 2 Summary")
    print("-" * 30)
    print("""
Key concepts covered:
1. Permutation invariance: The encoder output depends only on the content of the set.
2. Self-attention: Mechanisms that allow elements to interact without assuming sequence order.
3. Layer Normalization & Residuals: Standard components for training stability in attention models.
    """)
