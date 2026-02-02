"""
Solution 3: Encoder Stack
=========================

Complete solution for Exercise 3.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers (deep copies)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head attention (from Solution 2)."""
    
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalizes across the feature dimension (last dimension).
    """
    
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # gamma (scale)
        self.b_2 = nn.Parameter(torch.zeros(features))  # beta (shift)
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer norm.
    
    Implements: x + Dropout(Sublayer(LayerNorm(x)))
    """
    
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)    # Expand
        self.w_2 = nn.Linear(d_ff, d_model)    # Contract
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.
    
    Two sub-layers:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        # Self-attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Feed-forward sublayer
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Encoder: Stack of N encoder layers.
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# TESTS
# =============================================================================

def test_layer_norm():
    """Test LayerNorm implementation."""
    print("Testing LayerNorm...")
    
    ln = LayerNorm(features=64)
    
    x = torch.randn(2, 10, 64)
    out = ln(x)
    
    assert out.shape == x.shape
    
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=0.1)
    
    print("✓ LayerNorm test passed!")
    return True


def test_sublayer_connection():
    """Test SublayerConnection (residual + norm)."""
    print("\nTesting SublayerConnection...")
    
    size = 64
    slc = SublayerConnection(size, dropout=0.0)
    
    x = torch.randn(2, 10, size)
    
    out = slc(x, lambda x: x)
    assert out.shape == x.shape
    
    out_zero = slc(x, lambda x: torch.zeros_like(x))
    assert out_zero.shape == x.shape
    
    print("✓ SublayerConnection test passed!")
    return True


def test_feed_forward():
    """Test PositionwiseFeedForward."""
    print("\nTesting PositionwiseFeedForward...")
    
    d_model, d_ff = 64, 256
    ff = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    
    x = torch.randn(2, 10, d_model)
    out = ff(x)
    
    assert out.shape == x.shape
    
    expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    actual_params = sum(p.numel() for p in ff.parameters())
    assert actual_params == expected_params
    
    print(f"✓ FFN params: {actual_params:,}")
    print("✓ PositionwiseFeedForward test passed!")
    return True


def test_encoder_layer():
    """Test single EncoderLayer."""
    print("\nTesting EncoderLayer...")
    
    d_model, h, d_ff = 128, 4, 512
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    layer = EncoderLayer(d_model, attn, ff, dropout=0.1)
    
    x = torch.randn(2, 10, d_model)
    mask = torch.ones(2, 1, 10)
    
    out = layer(x, mask)
    
    assert out.shape == x.shape
    
    print(f"✓ EncoderLayer output: {out.shape}")
    print("✓ EncoderLayer test passed!")
    return True


def test_encoder():
    """Test full Encoder stack."""
    print("\nTesting Encoder stack...")
    
    d_model, h, d_ff, N = 128, 4, 512, 6
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout=0.1)
    encoder = Encoder(layer, N)
    
    x = torch.randn(2, 10, d_model)
    mask = torch.ones(2, 1, 10)
    
    out = encoder(x, mask)
    
    assert out.shape == x.shape
    assert len(encoder.layers) == N
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ Encoder layers: {N}")
    print(f"✓ Total params: {total_params:,}")
    print(f"✓ Output shape: {out.shape}")
    print("✓ Encoder test passed!")
    return True


def test_gradient_flow():
    """Test gradients flow through encoder."""
    print("\nTesting gradient flow...")
    
    d_model, h, d_ff, N = 64, 2, 128, 2
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout=0.0)
    encoder = Encoder(layer, N)
    
    x = torch.randn(2, 5, d_model, requires_grad=True)
    mask = torch.ones(2, 1, 5)
    
    out = encoder(x, mask)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ All parameters have gradients")
    print("✓ Gradient flow test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SOLUTION 3: ENCODER STACK")
    print("=" * 60)
    
    test_layer_norm()
    test_sublayer_connection()
    test_feed_forward()
    test_encoder_layer()
    test_encoder()
    test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
