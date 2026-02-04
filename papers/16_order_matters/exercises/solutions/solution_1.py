"""
Solution 1: Basic Pointer Attention Mechanism

Complete implementation with explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointerAttention(nn.Module):
    """
    Basic pointer attention: Given a query (decoder state), compute
    attention over keys (encoder outputs) to select an element.
    
    Formula: scores = v^T * tanh(W_key * keys + W_query * query)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Three linear layers for attention computation
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query, keys, mask=None):
        """
        Compute pointer attention.
        
        Args:
            query: [batch, hidden_dim] - Current decoder state
            keys: [batch, seq_len, hidden_dim] - Encoder outputs
            mask: [batch, seq_len] - 0 for positions to ignore, 1 for valid
            
        Returns:
            pointer: [batch] - Index of selected element
            attention_weights: [batch, seq_len] - Attention distribution
        """
        batch_size, seq_len, hidden_dim = keys.size()
        
        # Step 1: Project keys [batch, seq_len, hidden_dim]
        keys_proj = self.W_key(keys)
        
        # Step 2: Project query and expand [batch, 1, hidden_dim]
        query_proj = self.W_query(query).unsqueeze(1)
        
        # Step 3: Combine with tanh [batch, seq_len, hidden_dim]
        combined = torch.tanh(keys_proj + query_proj)
        
        # Step 4: Project to scalar scores [batch, seq_len]
        scores = self.v(combined).squeeze(-1)
        
        # Step 5: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 6: Softmax to get probabilities
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 7: Select pointer (argmax)
        pointer = torch.argmax(attention_weights, dim=-1)
        
        return pointer, attention_weights


def test_pointer_attention():
    """Test the pointer attention implementation."""
    print("🧪 Testing Pointer Attention")
    print("=" * 60)
    
    # Setup
    batch_size = 4
    seq_len = 5
    hidden_dim = 32
    
    # Create model
    attention = SimplePointerAttention(hidden_dim)
    
    # Create dummy data
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test 1: Basic forward pass
    print("\n✅ Test 1: Basic forward pass")
    pointer, weights = attention(query, keys)
    print(f"   Pointer shape: {pointer.shape} (expected: [{batch_size}])")
    print(f"   Weights shape: {weights.shape} (expected: [{batch_size}, {seq_len}])")
    print(f"   Example pointer: {pointer[0].item()}")
    print(f"   Example weights: {[f'{w:.3f}' for w in weights[0].tolist()]}")
    print(f"   Weights sum: {weights[0].sum().item():.4f} (should be ~1.0)")
    
    assert pointer.shape == (batch_size,), "Pointer shape incorrect"
    assert weights.shape == (batch_size, seq_len), "Weights shape incorrect"
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5), "Weights don't sum to 1"
    
    # Test 2: Masking
    print("\n✅ Test 2: Masking")
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0  # Mask last 2 positions
    
    pointer_masked, weights_masked = attention(query, keys, mask=mask)
    print(f"   Masked weights: {[f'{w:.3f}' for w in weights_masked[0].tolist()]}")
    print(f"   Last 2 weights: {weights_masked[0, -2:].tolist()} (should be ~0)")
    
    assert weights_masked[0, -2:].sum() < 0.01, "Masking not working"
    
    # Test 3: Attention concentrates on relevant keys
    print("\n✅ Test 3: Attention should concentrate")
    keys_biased = torch.randn(batch_size, seq_len, hidden_dim)
    keys_biased[:, 0] = query  # First key = query
    
    pointer_biased, weights_biased = attention(query, keys_biased)
    print(f"   Pointer: {pointer_biased[0].item()} (should be 0 or close)")
    print(f"   First weight: {weights_biased[0, 0].item():.4f} (should be highest)")
    print(f"   All weights: {[f'{w:.3f}' for w in weights_biased[0].tolist()]}")
    
    assert weights_biased[0, 0] > 0.5, "Attention not focusing on similar key"
    
    print("\n" + "=" * 60)
    print("✨ All tests passed!")


if __name__ == "__main__":
    test_pointer_attention()
    
    print("\n" + "=" * 60)
    print("🎯 Solution 1 Summary")
    print("=" * 60)
    print("""
Key implementation details:

1. **Projection layers**: Three nn.Linear layers transform inputs
   - W_key: Projects encoder outputs
   - W_query: Projects decoder state
   - v: Projects to scalar scores

2. **Attention formula**: score = v^T * tanh(W_key*keys + W_query*query)
   - Tanh creates bounded values [-1, 1]
   - Broadcasting handles batch dimension automatically

3. **Masking**: Set masked positions to -inf BEFORE softmax
   - softmax(-inf) = 0 (exactly what we want!)
   - Never mask AFTER softmax (won't work)

4. **Pointer selection**: Use argmax to pick highest attention
   - This is greedy selection
   - Could use sampling for exploration

Alternative implementations:
- Multiplicative attention: score = query^T * W * key (faster but less expressive)
- Additive attention: What we implemented (more parameters, more expressive)
- Dot-product attention: score = query^T * key / sqrt(d) (simplest, used in Transformers)
    """)
