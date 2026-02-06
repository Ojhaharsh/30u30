"""
Exercise 1: Basic Pointer Attention Mechanism
==============================================

Learn how to compute attention scores and "point" to input elements.

Concept: Given a decoder state and encoder outputs, compute which
         input element is most relevant right now.

[Our Addition: Analogy] Picking items from a menu -- you focus on one item at a time 
         based on your current preference (decoder state).

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
        keys_proj = None 
        
        # TODO: Step 2 - Project query and expand to match keys
        # Shape: [batch, 1, hidden_dim]
        query_proj = None 
        
        # TODO: Step 3 - Combine with tanh activation
        # Shape: [batch, seq_len, hidden_dim]
        combined = None 
        
        # TODO: Step 4 - Project to scalar scores
        # Shape: [batch, seq_len]
        scores = None 
        
        # TODO: Step 5 - Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # TODO: Step 6 - Compute attention weights with softmax
        attention_weights = None 
        
        # TODO: Step 7 - Select pointer (argmax)
        pointer = None 
        
        return pointer, attention_weights


def test_pointer_attention():
    """Test the pointer attention implementation."""
    print("Testing Pointer Attention")
    print("-" * 30)
    
    batch_size = 4
    seq_len = 5
    hidden_dim = 32
    
    attention = SimplePointerAttention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test 1: Basic forward pass
    print("Test 1: Basic forward pass")
    pointer, weights = attention(query, keys)
    if pointer is not None:
        print(f"   Pointer shape: {pointer.shape}")
        print(f"   Weights shape: {weights.shape}")
        print(f"   Weights sum: {weights[0].sum().item():.4f}")
    
    # Test 2: Masking
    print("Test 2: Masking")
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0
    pointer_masked, weights_masked = attention(query, keys, mask=mask)
    if weights_masked is not None:
        print(f"   Last 2 weights: {weights_masked[0, -2:].tolist()} (should be 0.0)")
    
    # Test 3: Numerical focus
    print("Test 3: Attention should concentrate")
    keys_biased = torch.randn(batch_size, seq_len, hidden_dim)
    keys_biased[:, 0] = query
    pointer_biased, weights_biased = attention(query, keys_biased)
    if weights_biased is not None:
        print(f"   First weight: {weights_biased[0, 0].item():.4f} (should be high)")

    print("-" * 30)


if __name__ == "__main__":
    test_pointer_attention()
    
    print("Exercise 1 Summary")
    print("-" * 30)
    print("""
Key concepts covered:
1. Attention scores: Measuring relevance between query and keys.
2. Softmax: Converting scores to a probability distribution.
3. Argmax: Selecting the most relevant index.
4. Masking: Ensuring invalid positions are ignored.
    """)
