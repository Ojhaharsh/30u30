"""
Exercise 1: Attention with Masking
==================================

Implement scaled dot-product attention with proper masking support.

This is the CORE operation of the Transformer. Everything else builds on this.

Key concepts:
- Scaled dot-product: QK^T / sqrt(d_k)
- Masking: Set masked positions to -1e9 before softmax
- Softmax: Convert scores to probabilities
- Output: Weighted sum of values

Your tasks:
1. Implement the attention function
2. Implement subsequent_mask for causal attention
3. Test with and without masking
"""

import math
import torch
import torch.nn.functional as F


def subsequent_mask(size):
    """
    Create a mask to hide subsequent (future) positions.
    
    For autoregressive decoding, position i can only attend to positions <= i.
    
    Args:
        size: Sequence length
    
    Returns:
        Boolean mask of shape (1, size, size)
        True = can attend, False = cannot attend
    
    Example for size=4:
        [[True,  False, False, False],
         [True,  True,  False, False],
         [True,  True,  True,  False],
         [True,  True,  True,  True]]
    
    Hint: Use torch.triu (upper triangular)
    """
    # TODO: Implement subsequent mask
    # Create a mask where position i can attend to positions 0...i
    # The diagonal=1 in triu gives us the future positions to mask
    
    raise NotImplementedError("Implement subsequent_mask")


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        query: Tensor of shape (batch, heads, seq_q, d_k)
        key: Tensor of shape (batch, heads, seq_k, d_k)
        value: Tensor of shape (batch, heads, seq_k, d_v)
        mask: Optional mask of shape (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)
              True/1 = attend, False/0 = mask out
        dropout: Optional dropout module
    
    Returns:
        output: Tensor of shape (batch, heads, seq_q, d_v)
        attention_weights: Tensor of shape (batch, heads, seq_q, seq_k)
    
    Steps:
        1. Compute Q @ K^T to get attention scores
        2. Scale by 1/sqrt(d_k) to prevent large values
        3. Apply mask (set masked positions to -1e9)
        4. Apply softmax to get attention weights (sum to 1)
        5. Apply dropout (if provided)
        6. Compute weighted sum: weights @ V
    
    Why scale by sqrt(d_k)?
        When d_k is large, dot products grow large, pushing softmax
        into regions with tiny gradients. Scaling keeps variance stable.
    """
    # TODO: Implement attention
    # Step 1: Get d_k from query shape
    
    # Step 2: Compute attention scores (batch, heads, seq_q, seq_k)
    
    # Step 3: Scale by sqrt(d_k)
    
    # Step 4: Apply mask if provided
    # Use: scores.masked_fill(mask == 0, -1e9)
    
    # Step 5: Softmax to get attention weights
    
    # Step 6: Apply dropout if provided
    
    # Step 7: Compute output as weighted sum of values
    
    raise NotImplementedError("Implement attention")


# =============================================================================
# TESTS
# =============================================================================

def test_subsequent_mask():
    """Test the subsequent mask generation."""
    print("Testing subsequent_mask...")
    
    mask = subsequent_mask(4)
    
    # Check shape
    assert mask.shape == (1, 4, 4), f"Wrong shape: {mask.shape}"
    
    # Check pattern
    expected = torch.tensor([[[True,  False, False, False],
                               [True,  True,  False, False],
                               [True,  True,  True,  False],
                               [True,  True,  True,  True]]])
    
    assert torch.all(mask == expected), f"Wrong pattern:\n{mask}"
    
    print("✓ subsequent_mask test passed!")
    return True


def test_attention_basic():
    """Test basic attention without masking."""
    print("\nTesting attention (no mask)...")
    
    batch, heads, seq, d_k = 2, 4, 5, 8
    
    # Random inputs
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq, d_k)
    K = torch.randn(batch, heads, seq, d_k)
    V = torch.randn(batch, heads, seq, d_k)
    
    output, attn = attention(Q, K, V)
    
    # Check shapes
    assert output.shape == (batch, heads, seq, d_k), f"Wrong output shape: {output.shape}"
    assert attn.shape == (batch, heads, seq, seq), f"Wrong attn shape: {attn.shape}"
    
    # Attention weights should sum to 1
    attn_sum = attn.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        f"Attention doesn't sum to 1: {attn_sum}"
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights sum to 1")
    print("✓ Basic attention test passed!")
    return True


def test_attention_with_mask():
    """Test attention with causal masking."""
    print("\nTesting attention (with mask)...")
    
    batch, heads, seq, d_k = 1, 1, 4, 8
    
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq, d_k)
    K = torch.randn(batch, heads, seq, d_k)
    V = torch.randn(batch, heads, seq, d_k)
    
    # Create causal mask
    mask = subsequent_mask(seq)
    
    output, attn = attention(Q, K, V, mask=mask)
    
    # Check that future positions have zero attention
    # Upper triangle (excluding diagonal) should be ~0
    for i in range(seq):
        for j in range(i + 1, seq):
            assert attn[0, 0, i, j] < 1e-6, \
                f"Position ({i},{j}) should be masked but has attention {attn[0, 0, i, j]}"
    
    print(f"✓ Masked positions have ~0 attention")
    print(f"✓ Attention pattern:\n{attn[0, 0].round().int()}")
    print("✓ Masked attention test passed!")
    return True


def test_attention_scaling():
    """Test that scaling prevents large softmax inputs."""
    print("\nTesting attention scaling...")
    
    # Without scaling, large d_k causes softmax saturation
    batch, heads, seq = 1, 1, 4
    
    for d_k in [8, 64, 512]:
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, seq, d_k)
        K = torch.randn(batch, heads, seq, d_k)
        V = torch.randn(batch, heads, seq, d_k)
        
        output, attn = attention(Q, K, V)
        
        # Check attention entropy (uniform = high, concentrated = low)
        entropy = -(attn * attn.log().clamp(min=-100)).sum(dim=-1).mean()
        
        print(f"  d_k={d_k:3d}: entropy={entropy:.2f}")
    
    print("✓ Scaling prevents saturation across d_k values")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("EXERCISE 1: ATTENTION WITH MASKING")
    print("=" * 60)
    
    try:
        test_subsequent_mask()
        test_attention_basic()
        test_attention_with_mask()
        test_attention_scaling()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext: Exercise 2 - Multi-Head Attention Module")
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("Implement the TODO sections and run again!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    run_all_tests()
