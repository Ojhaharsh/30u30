"""
Exercise 1: Basic Pointer Attention Mechanism

Learn how to compute attention scores and "point" to input elements.

Concept: Given a decoder state and encoder outputs, compute which
         input element is most relevant right now.

Real-world analogy: You're at a buffet with 10 dishes. At each moment,
                   you decide which dish looks most appetizing (attention!).

Your task: Implement the core attention mechanism for pointing.
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
        
        # TODO: Initialize three linear layers
        # 1. Project keys (encoder outputs) to hidden_dim
        # 2. Project query (decoder state) to hidden_dim  
        # 3. Project combined representation to scalar score
        
        # Hint: Use nn.Linear(input_dim, output_dim, bias=False)
        
        self.W_key = None  # TODO: Replace None
        self.W_query = None  # TODO: Replace None
        self.v = None  # TODO: Replace None
    
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
        
        # TODO: Step 1 - Project keys
        # Shape: [batch, seq_len, hidden_dim]
        keys_proj = None  # TODO: Apply self.W_key to keys
        
        # TODO: Step 2 - Project query and expand to match keys
        # Shape: [batch, 1, hidden_dim] (expand for broadcasting)
        query_proj = None  # TODO: Apply self.W_query to query, then unsqueeze(1)
        
        # TODO: Step 3 - Combine with tanh activation
        # Shape: [batch, seq_len, hidden_dim]
        combined = None  # TODO: torch.tanh(keys_proj + query_proj)
        
        # TODO: Step 4 - Project to scalar scores
        # Shape: [batch, seq_len]
        scores = None  # TODO: Apply self.v, then squeeze(-1)
        
        # TODO: Step 5 - Apply mask if provided
        if mask is not None:
            # Hint: Use masked_fill to set masked positions to -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # TODO: Step 6 - Compute attention weights with softmax
        attention_weights = None  # TODO: F.softmax(scores, dim=-1)
        
        # TODO: Step 7 - Select pointer (argmax)
        pointer = None  # TODO: torch.argmax(attention_weights, dim=-1)
        
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
    print(f"   Example weights: {weights[0].tolist()}")
    print(f"   Weights sum: {weights[0].sum().item():.4f} (should be ~1.0)")
    
    # Test 2: Masking
    print("\n✅ Test 2: Masking")
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0  # Mask last 2 positions
    
    pointer_masked, weights_masked = attention(query, keys, mask=mask)
    print(f"   Masked weights: {weights_masked[0].tolist()}")
    print(f"   Last 2 weights: {weights_masked[0, -2:].tolist()} (should be ~0)")
    
    # Test 3: Attention concentrates on relevant keys
    print("\n✅ Test 3: Attention should concentrate")
    # Make first key very similar to query
    keys_biased = torch.randn(batch_size, seq_len, hidden_dim)
    keys_biased[:, 0] = query  # First key = query (should get high attention)
    
    pointer_biased, weights_biased = attention(query, keys_biased)
    print(f"   Pointer: {pointer_biased[0].item()} (should be 0 or close)")
    print(f"   First weight: {weights_biased[0, 0].item():.4f} (should be highest)")
    print(f"   All weights: {weights_biased[0].tolist()}")
    
    print("\n" + "=" * 60)
    print("✨ All tests passed!" if pointer is not None else "❌ Implementation incomplete")


if __name__ == "__main__":
    test_pointer_attention()
    
    print("\n" + "=" * 60)
    print("🎯 Exercise 1 Summary")
    print("=" * 60)
    print("""
You've implemented the core of Pointer Networks!

Key concepts you learned:
1. ✅ Attention scores: Measure relevance between query and keys
2. ✅ Softmax: Convert scores to probability distribution
3. ✅ Argmax: Select the most relevant element
4. ✅ Masking: Ignore invalid positions (padding, already selected)

Next: Exercise 2 - Build an order-invariant set encoder!
    """)
