"""
Solution 1: Basic Pointer Attention Mechanism
==============================================

Complete implementation of the Pointer Network attention mechanism.
Maps to Equation 1 in Vinyals et al. (2015).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointerAttention(nn.Module):
    """
    Pointer attention: Given a query (decoder state) and keys (encoder outputs),
    outputs a distribution over the input indices.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query, keys, mask=None):
        """
        Compute pointer attention.
        
        Args:
            query: [batch, hidden_dim]
            keys: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]
        """
        batch_size, seq_len, hidden_dim = keys.size()
        
        # Step 1: Project keys
        keys_proj = self.W_key(keys)
        
        # Step 2: Project query
        query_proj = self.W_query(query).unsqueeze(1)
        
        # Step 3: Energy computation (Eq 1)
        # Using tanh activation as defined in the paper
        combined = torch.tanh(keys_proj + query_proj)
        
        # Step 4: Map to scores
        scores = self.v(combined).squeeze(-1)
        
        # Step 5: Masking (sampling without replacement)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 6: Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 7: Selection
        pointer = torch.argmax(attention_weights, dim=-1)
        
        return pointer, attention_weights


def test_pointer_attention():
    print("Validating Pointer Attention Solution")
    print("-" * 30)
    
    batch_size = 4
    seq_len = 5
    hidden_dim = 32
    
    attention = SimplePointerAttention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    
    pointer, weights = attention(query, keys)
    assert pointer.shape == (batch_size,)
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))
    print("Forward pass verified")
    
    mask = torch.ones(batch_size, seq_len)
    mask[:, -1] = 0
    _, weights_masked = attention(query, keys, mask=mask)
    assert weights_masked[0, -1] < 1e-7
    print("Masking verified")
    
    print("All validation tests complete")

if __name__ == "__main__":
    test_pointer_attention()
