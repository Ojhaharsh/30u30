"""
Exercise 1: Implement Basic Additive (Bahdanau) Attention

The core of Bahdanau attention is the alignment model:
    e_{t,i} = v^T * tanh(W_s * s + W_h * h)
    α_{t,i} = softmax(e_{t,i})
    c_t = Σ α_{t,i} * h_i

Your task: Implement the attention mechanism from scratch.

Think of it like this:
- You're in a library (encoder outputs = books on shelves)
- You have a question in mind (query = decoder state)
- You need to figure out which books are relevant (attention weights)
- Then you read the relevant parts (context = weighted sum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    TODO: Implement additive (Bahdanau) attention.
    
    The attention computes:
    1. Project query: W_s * s (where s is the decoder state)
    2. Project keys: W_h * h (where h are encoder outputs)
    3. Combine: tanh(projected_query + projected_keys)
    4. Score: v^T * combined (scalar for each position)
    5. Normalize: softmax over positions
    6. Context: weighted sum of values (encoder outputs)
    
    Args:
        hidden_size: Dimension of the attention hidden layer
        query_size: Dimension of the query (decoder hidden state)
        key_size: Dimension of keys/values (encoder outputs)
    """
    
    def __init__(self, hidden_size: int, query_size: int, key_size: int):
        super().__init__()
        
        # TODO: Create three linear layers:
        # 1. query_proj: projects query from query_size to hidden_size
        # 2. key_proj: projects keys from key_size to hidden_size
        # 3. energy: projects from hidden_size to 1 (scalar score)
        
        self.query_proj = None  # TODO
        self.key_proj = None    # TODO
        self.energy = None      # TODO
        
        raise NotImplementedError("Implement the __init__ method!")
    
    def forward(
        self, 
        query: torch.Tensor,    # [batch, query_size]
        keys: torch.Tensor,     # [batch, seq_len, key_size]
        values: torch.Tensor,   # [batch, seq_len, value_size]
        mask: torch.Tensor = None  # [batch, seq_len] True = ignore
    ):
        """
        Compute attention and return context vector.
        
        Returns:
            context: [batch, value_size]
            weights: [batch, seq_len]
        """
        # TODO: Implement the forward pass
        # Step 1: Project query -> [batch, 1, hidden_size]
        # Step 2: Project keys -> [batch, seq_len, hidden_size]
        # Step 3: Add and apply tanh -> [batch, seq_len, hidden_size]
        # Step 4: Compute energy scores -> [batch, seq_len]
        # Step 5: Apply mask if provided (set masked to -inf)
        # Step 6: Softmax to get attention weights -> [batch, seq_len]
        # Step 7: Compute context as weighted sum of values -> [batch, value_size]
        
        raise NotImplementedError("Implement the forward method!")


# ============================================================================
# Tests
# ============================================================================

def test_attention():
    """Test your implementation."""
    print("Testing Additive Attention...")
    
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    query_size = 128
    key_size = 256
    
    # Create attention module
    attention = AdditiveAttention(
        hidden_size=hidden_size,
        query_size=query_size,
        key_size=key_size
    )
    
    # Create dummy inputs
    query = torch.randn(batch_size, query_size)
    keys = torch.randn(batch_size, seq_len, key_size)
    values = keys  # In Bahdanau, keys and values are the same (encoder outputs)
    
    # Forward pass
    context, weights = attention(query, keys, values)
    
    # Check shapes
    assert context.shape == (batch_size, key_size), \
        f"Context shape wrong: {context.shape}"
    assert weights.shape == (batch_size, seq_len), \
        f"Weights shape wrong: {weights.shape}"
    
    # Check attention weights sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sums}"
    
    # Test with mask
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, -2:] = True  # Mask last 2 positions
    
    context_masked, weights_masked = attention(query, keys, values, mask)
    
    # Masked positions should have ~0 attention
    assert (weights_masked[:, -2:] < 0.01).all(), \
        f"Masked positions have too much attention: {weights_masked[:, -2:]}"
    
    print("✅ All tests passed!")
    print(f"   Context shape: {context.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights sum: {weights[0].sum().item():.4f}")


if __name__ == '__main__':
    test_attention()
