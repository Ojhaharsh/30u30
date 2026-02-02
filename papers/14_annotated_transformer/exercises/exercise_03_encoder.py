"""
Exercise 3: Encoder Stack
=========================

Build the complete encoder with proper layer composition.

The encoder transforms the input sequence into a rich representation
that captures relationships between all positions.

Encoder architecture:
1. Input embedding + positional encoding
2. N identical encoder layers, each with:
   - Multi-head self-attention + Add & Norm
   - Feed-forward network + Add & Norm
3. Final layer normalization

Key concepts:
- Residual connections (prevent vanishing gradients)
- Pre-normalization (normalize before sublayer)
- Layer stacking with clones()

Your tasks:
1. Implement LayerNorm
2. Implement SublayerConnection (residual + norm)
3. Implement PositionwiseFeedForward
4. Implement EncoderLayer
5. Implement Encoder (stack of layers)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers (deep copies)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# We'll use the MultiHeadedAttention from Exercise 2
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
    """Multi-head attention (provided from Exercise 2)."""
    
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
    Unlike BatchNorm, works the same during training and inference.
    
    LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta
    
    Args:
        features: Number of features (d_model)
        eps: Small constant for numerical stability
    
    Parameters:
        a_2 (gamma): Learned scale parameter, initialized to 1
        b_2 (beta): Learned shift parameter, initialized to 0
    """
    
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # TODO: Create learnable parameters
        # self.a_2 = nn.Parameter(torch.ones(features))   # gamma
        # self.b_2 = nn.Parameter(torch.zeros(features))  # beta
        # self.eps = eps
        
        raise NotImplementedError("Implement LayerNorm.__init__")
    
    def forward(self, x):
        """
        Apply layer normalization.
        
        Steps:
        1. Compute mean across last dimension
        2. Compute std across last dimension
        3. Normalize: (x - mean) / (std + eps)
        4. Scale and shift: gamma * normalized + beta
        """
        # TODO: Implement normalization
        # mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
        raise NotImplementedError("Implement LayerNorm.forward")


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer norm.
    
    This implements: x + Dropout(Sublayer(LayerNorm(x)))
    
    Note: This uses PRE-NORM (normalize before sublayer), which is
    more stable for training deep networks than POST-NORM.
    
    Args:
        size: Model dimension (d_model)
        dropout: Dropout rate
    """
    
    def __init__(self, size, dropout):
        super().__init__()
        # TODO: Create LayerNorm and Dropout
        # self.norm = LayerNorm(size)
        # self.dropout = nn.Dropout(dropout)
        
        raise NotImplementedError("Implement SublayerConnection.__init__")
    
    def forward(self, x, sublayer):
        """
        Apply residual connection.
        
        Args:
            x: Input tensor
            sublayer: Function to apply (e.g., attention or FFN)
        
        The pattern: x + dropout(sublayer(norm(x)))
        """
        # TODO: Implement pre-norm residual connection
        # return x + self.dropout(sublayer(self.norm(x)))
        
        raise NotImplementedError("Implement SublayerConnection.forward")


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Two fully connected layers with ReLU activation in between.
    Applied to each position independently (hence "position-wise").
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Create two linear layers and dropout
        # self.w_1 = nn.Linear(d_model, d_ff)    # Expand
        # self.w_2 = nn.Linear(d_ff, d_model)    # Contract
        # self.dropout = nn.Dropout(dropout)
        
        raise NotImplementedError("Implement PositionwiseFeedForward.__init__")
    
    def forward(self, x):
        """
        Apply feed-forward network.
        
        Steps:
        1. First linear: d_model -> d_ff
        2. ReLU activation
        3. Dropout
        4. Second linear: d_ff -> d_model
        """
        # TODO: Implement FFN
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
        raise NotImplementedError("Implement PositionwiseFeedForward.forward")


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.
    
    Each layer has two sub-layers:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    
    Each sub-layer has a residual connection and layer normalization.
    
    Args:
        size: Model dimension (d_model)
        self_attn: MultiHeadedAttention module
        feed_forward: PositionwiseFeedForward module
        dropout: Dropout rate
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        # TODO: Store modules and create sublayer connections
        # self.self_attn = self_attn
        # self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # self.size = size
        
        raise NotImplementedError("Implement EncoderLayer.__init__")
    
    def forward(self, x, mask):
        """
        Apply encoder layer.
        
        Args:
            x: Input (batch, seq, d_model)
            mask: Source mask (batch, 1, seq)
        
        Steps:
        1. Self-attention with residual: x = sublayer[0](x, self_attn)
        2. Feed-forward with residual: x = sublayer[1](x, feed_forward)
        
        Note: For self-attention, Q=K=V=x
        """
        # TODO: Apply self-attention sublayer
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # TODO: Apply feed-forward sublayer
        # return self.sublayer[1](x, self.feed_forward)
        
        raise NotImplementedError("Implement EncoderLayer.forward")


class Encoder(nn.Module):
    """
    Encoder: Stack of N encoder layers.
    
    The encoder maps an input sequence to a continuous representation
    that the decoder will attend to.
    
    Args:
        layer: EncoderLayer to clone
        N: Number of layers
    """
    
    def __init__(self, layer, N):
        super().__init__()
        # TODO: Clone layers and create final norm
        # self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)
        
        raise NotImplementedError("Implement Encoder.__init__")
    
    def forward(self, x, mask):
        """
        Pass input through all layers, then normalize.
        
        Args:
            x: Input (batch, seq, d_model)
            mask: Source mask
        
        Returns:
            Encoded representation (batch, seq, d_model)
        """
        # TODO: Pass through all layers, then final norm
        # for layer in self.layers:
        #     x = layer(x, mask)
        # return self.norm(x)
        
        raise NotImplementedError("Implement Encoder.forward")


# =============================================================================
# TESTS
# =============================================================================

def test_layer_norm():
    """Test LayerNorm implementation."""
    print("Testing LayerNorm...")
    
    ln = LayerNorm(features=64)
    
    x = torch.randn(2, 10, 64)
    out = ln(x)
    
    # Check shape preserved
    assert out.shape == x.shape, f"Shape changed: {x.shape} -> {out.shape}"
    
    # Check normalized (mean ~0, std ~1 for each position)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
        f"Mean not ~0: {mean.abs().max()}"
    assert torch.allclose(std, torch.ones_like(std), atol=0.1), \
        f"Std not ~1: {std}"
    
    print("✓ LayerNorm test passed!")
    return True


def test_sublayer_connection():
    """Test SublayerConnection (residual + norm)."""
    print("\nTesting SublayerConnection...")
    
    size = 64
    slc = SublayerConnection(size, dropout=0.0)  # No dropout for testing
    
    x = torch.randn(2, 10, size)
    
    # Test with identity sublayer
    out = slc(x, lambda x: x)
    
    # With identity, output should be x + x = 2x (approximately, after norm)
    assert out.shape == x.shape
    
    # Test with zero sublayer (output should be ~x after norm)
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
    
    assert out.shape == x.shape, f"Shape changed: {x.shape} -> {out.shape}"
    
    # Check parameter count
    expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    actual_params = sum(p.numel() for p in ff.parameters())
    assert actual_params == expected_params, \
        f"Wrong param count: {actual_params} vs {expected_params}"
    
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
    
    assert out.shape == x.shape, f"Shape changed: {x.shape} -> {out.shape}"
    
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
    
    assert out.shape == x.shape, f"Shape changed: {x.shape} -> {out.shape}"
    
    # Count layers
    assert len(encoder.layers) == N, f"Wrong layer count: {len(encoder.layers)}"
    
    # Count parameters
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
    
    assert x.grad is not None, "No gradient for input"
    assert x.grad.abs().sum() > 0, "Zero gradients"
    
    # Check all parameters have gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ All parameters have gradients")
    print("✓ Gradient flow test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("EXERCISE 3: ENCODER STACK")
    print("=" * 60)
    
    try:
        test_layer_norm()
        test_sublayer_connection()
        test_feed_forward()
        test_encoder_layer()
        test_encoder()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext: Exercise 4 - Training Pipeline")
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("Implement the TODO sections and run again!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    run_all_tests()
