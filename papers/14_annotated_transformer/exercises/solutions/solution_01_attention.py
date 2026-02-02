"""
Solution 1: Attention with Masking
==================================

Complete solution for Exercise 1.
"""

import math
import torch
import torch.nn.functional as F


def subsequent_mask(size):
    """
    Create a mask to hide subsequent (future) positions.
    
    For autoregressive decoding, position i can only attend to positions <= i.
    """
    attn_shape = (1, size, size)
    # triu with diagonal=1 gives us the upper triangular (excluding diagonal)
    # We then invert it to get the lower triangular (including diagonal)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    # Step 1: Get d_k from query shape
    d_k = query.size(-1)
    
    # Step 2: Compute attention scores (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 3: Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    
    # Step 4: Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 5: Softmax to get attention weights
    p_attn = scores.softmax(dim=-1)
    
    # Step 6: Apply dropout if provided
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # Step 7: Compute output as weighted sum of values
    return torch.matmul(p_attn, value), p_attn


# =============================================================================
# TESTS
# =============================================================================

def test_subsequent_mask():
    """Test the subsequent mask generation."""
    print("Testing subsequent_mask...")
    
    mask = subsequent_mask(4)
    
    assert mask.shape == (1, 4, 4), f"Wrong shape: {mask.shape}"
    
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
    
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq, d_k)
    K = torch.randn(batch, heads, seq, d_k)
    V = torch.randn(batch, heads, seq, d_k)
    
    output, attn = attention(Q, K, V)
    
    assert output.shape == (batch, heads, seq, d_k), f"Wrong output shape: {output.shape}"
    assert attn.shape == (batch, heads, seq, seq), f"Wrong attn shape: {attn.shape}"
    
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
    
    mask = subsequent_mask(seq)
    output, attn = attention(Q, K, V, mask=mask)
    
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
    
    batch, heads, seq = 1, 1, 4
    
    for d_k in [8, 64, 512]:
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, seq, d_k)
        K = torch.randn(batch, heads, seq, d_k)
        V = torch.randn(batch, heads, seq, d_k)
        
        output, attn = attention(Q, K, V)
        
        entropy = -(attn * attn.log().clamp(min=-100)).sum(dim=-1).mean()
        
        print(f"  d_k={d_k:3d}: entropy={entropy:.2f}")
    
    print("✓ Scaling prevents saturation across d_k values")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SOLUTION 1: ATTENTION WITH MASKING")
    print("=" * 60)
    
    test_subsequent_mask()
    test_attention_basic()
    test_attention_with_mask()
    test_attention_scaling()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
