"""
Solution 2: Multi-Head Attention Module
========================================

Complete solution for Exercise 2.
"""

import math
import copy
import torch
import torch.nn as nn


def clones(module, N):
    """Produce N identical layers (deep copies)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention Module.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        
        # Check divisibility
        assert d_model % h == 0, "d_model must be divisible by h"
        
        # Calculate dimension per head
        self.d_k = d_model // h
        self.h = h
        
        # Four linear projections: Q, K, V, and output
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
        # Store attention weights for visualization
        self.attn = None
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        """Apply multi-head attention."""
        
        # Handle mask dimension (broadcast over heads)
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # Get batch size
        nbatches = query.size(0)
        
        # 1) Project and reshape: (batch, seq, d_model) -> (batch, h, seq, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]
        
        # 2) Apply attention to all heads in parallel
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) Concatenate heads: (batch, h, seq, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        # 4) Final linear projection
        return self.linears[-1](x)


# =============================================================================
# TESTS
# =============================================================================

def test_init():
    """Test module initialization."""
    print("Testing MultiHeadedAttention initialization...")
    
    mha = MultiHeadedAttention(h=8, d_model=512, dropout=0.1)
    
    assert hasattr(mha, 'd_k'), "Missing d_k attribute"
    assert hasattr(mha, 'h'), "Missing h attribute"
    assert hasattr(mha, 'linears'), "Missing linears attribute"
    assert hasattr(mha, 'dropout'), "Missing dropout attribute"
    
    assert mha.d_k == 64, f"d_k should be 64, got {mha.d_k}"
    assert mha.h == 8, f"h should be 8, got {mha.h}"
    assert len(mha.linears) == 4, f"Should have 4 linears, got {len(mha.linears)}"
    
    print("✓ Initialization test passed!")
    return True


def test_forward_shape():
    """Test output shapes."""
    print("\nTesting forward pass shapes...")
    
    batch, seq, d_model, h = 4, 10, 512, 8
    
    mha = MultiHeadedAttention(h=h, d_model=d_model)
    
    q = torch.randn(batch, seq, d_model)
    k = torch.randn(batch, seq, d_model)
    v = torch.randn(batch, seq, d_model)
    
    output = mha(q, k, v)
    
    assert output.shape == (batch, seq, d_model), \
        f"Wrong output shape: {output.shape}"
    
    print(f"✓ Output shape: {output.shape}")
    print("✓ Forward shape test passed!")
    return True


def test_attention_weights():
    """Test that attention weights are stored and valid."""
    print("\nTesting attention weights...")
    
    batch, seq, d_model, h = 2, 6, 128, 4
    
    mha = MultiHeadedAttention(h=h, d_model=d_model)
    
    q = torch.randn(batch, seq, d_model)
    k = torch.randn(batch, seq, d_model)
    v = torch.randn(batch, seq, d_model)
    
    _ = mha(q, k, v)
    
    assert mha.attn is not None, "Attention weights not stored"
    assert mha.attn.shape == (batch, h, seq, seq), \
        f"Wrong attn shape: {mha.attn.shape}"
    
    attn_sum = mha.attn.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        "Attention weights don't sum to 1"
    
    print(f"✓ Attention shape: {mha.attn.shape}")
    print("✓ Attention weights sum to 1")
    print("✓ Attention weights test passed!")
    return True


def test_with_mask():
    """Test with causal masking."""
    print("\nTesting with causal mask...")
    
    batch, seq, d_model, h = 2, 5, 64, 2
    
    mha = MultiHeadedAttention(h=h, d_model=d_model)
    
    q = torch.randn(batch, seq, d_model)
    k = torch.randn(batch, seq, d_model)
    v = torch.randn(batch, seq, d_model)
    
    mask = torch.triu(torch.ones(1, seq, seq), diagonal=1) == 0
    
    output = mha(q, k, v, mask=mask)
    
    for i in range(seq):
        for j in range(i + 1, seq):
            assert mha.attn[0, 0, i, j] < 1e-6, \
                f"Position ({i},{j}) should be masked"
    
    print("✓ Future positions correctly masked")
    print("✓ Mask test passed!")
    return True


def test_different_heads():
    """Test with different numbers of heads."""
    print("\nTesting different head counts...")
    
    batch, seq, d_model = 2, 8, 256
    
    for h in [1, 2, 4, 8, 16]:
        mha = MultiHeadedAttention(h=h, d_model=d_model)
        
        q = torch.randn(batch, seq, d_model)
        k = torch.randn(batch, seq, d_model)
        v = torch.randn(batch, seq, d_model)
        
        output = mha(q, k, v)
        
        assert output.shape == (batch, seq, d_model), \
            f"Wrong shape with h={h}"
        
        print(f"  h={h:2d}, d_k={d_model // h:3d}: ✓")
    
    print("✓ All head configurations work!")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\nTesting gradient flow...")
    
    mha = MultiHeadedAttention(h=4, d_model=128)
    
    q = torch.randn(2, 5, 128, requires_grad=True)
    k = torch.randn(2, 5, 128, requires_grad=True)
    v = torch.randn(2, 5, 128, requires_grad=True)
    
    output = mha(q, k, v)
    loss = output.sum()
    loss.backward()
    
    assert q.grad is not None, "No gradient for query"
    assert k.grad is not None, "No gradient for key"
    assert v.grad is not None, "No gradient for value"
    
    assert q.grad.abs().sum() > 0, "Zero gradients for query"
    
    print("✓ Gradients flow correctly")
    print("✓ Gradient test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SOLUTION 2: MULTI-HEAD ATTENTION MODULE")
    print("=" * 60)
    
    test_init()
    test_forward_shape()
    test_attention_weights()
    test_with_mask()
    test_different_heads()
    test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
