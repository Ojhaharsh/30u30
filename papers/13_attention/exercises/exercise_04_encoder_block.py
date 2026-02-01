"""
Exercise 4: Encoder Block

Build a complete Transformer encoder block with:
- Multi-head self-attention
- Residual connections
- Layer normalization
- Feed-forward network

Structure:
    x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


# Use reference implementations (or import from solution)
def multi_head_attention(query, key, value, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """Reference multi-head attention."""
    batch_size, seq_q, d_model = query.shape
    seq_k = key.shape[1]
    d_k = d_model // n_heads
    
    Q = query @ W_Q
    K = key @ W_K
    V = value @ W_V
    
    Q = Q.reshape(batch_size, seq_q, n_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_k, n_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_k, n_heads, d_k).transpose(0, 2, 1, 3)
    
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    attn = np.matmul(weights, V)
    
    attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, d_model)
    return attn @ W_O


class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes across the feature dimension (last axis).
    Unlike BatchNorm, works on single examples and is position-independent.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input (..., d_model)
            
        Returns:
            Normalized output (same shape)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    
    The hidden dimension is typically 4x the model dimension.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: Linear -> ReLU -> Linear."""
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class EncoderBlock:
    """
    Transformer Encoder Block.
    
    Components:
    1. Multi-head self-attention
    2. Residual connection + layer norm
    3. Feed-forward network
    4. Residual connection + layer norm
    
    TODO: Implement the forward method.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Attention weights
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # Layer norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed-forward
        self.ff = FeedForward(d_model, d_ff)
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional padding mask
            
        Returns:
            output: (batch, seq_len, d_model)
        
        Structure (Post-Norm, as in original paper):
            1. attn = MultiHeadAttention(x, x, x)
            2. x = LayerNorm(x + attn)
            3. ff = FeedForward(x)
            4. x = LayerNorm(x + ff)
        """
        # ==========================================
        # TODO: Implement encoder block forward pass
        # ==========================================
        
        # Step 1: Self-attention
        # attn_output = multi_head_attention(
        #     x, x, x, 
        #     self.W_Q, self.W_K, self.W_V, self.W_O,
        #     self.n_heads, mask
        # )
        
        # Step 2: Residual + Layer Norm
        # x = self.norm1.forward(x + attn_output)
        
        # Step 3: Feed-forward
        # ff_output = self.ff.forward(x)
        
        # Step 4: Residual + Layer Norm
        # x = self.norm2.forward(x + ff_output)
        
        # return x
        
        raise NotImplementedError("Implement encoder block forward pass!")


# =============================================================================
# TESTS
# =============================================================================

def test_layer_norm():
    """Test layer normalization."""
    ln = LayerNorm(64)
    x = np.random.randn(2, 10, 64) * 5 + 3  # Non-zero mean and variance
    
    output = ln.forward(x)
    
    # Output should have mean ~0 and std ~1 along last axis
    mean = output.mean(axis=-1)
    std = output.std(axis=-1)
    
    assert np.allclose(mean, 0, atol=0.1), f"Mean should be ~0, got {mean.mean()}"
    assert np.allclose(std, 1, atol=0.1), f"Std should be ~1, got {std.mean()}"
    
    print("[PASS] Layer norm test passed!")


def test_feed_forward():
    """Test feed-forward network."""
    ff = FeedForward(64, 256)
    x = np.random.randn(2, 10, 64)
    
    output = ff.forward(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    print("[PASS] Feed-forward test passed!")


def test_encoder_shape():
    """Test encoder block output shape."""
    encoder = EncoderBlock(d_model=64, n_heads=4, d_ff=256)
    x = np.random.randn(2, 10, 64)
    
    output = encoder.forward(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    print("[PASS] Encoder shape test passed!")


def test_residual_connection():
    """Test that residual connections are working."""
    np.random.seed(42)
    
    encoder = EncoderBlock(d_model=64, n_heads=4, d_ff=256)
    x = np.random.randn(1, 3, 64)
    
    output = encoder.forward(x)
    
    # With residual connections, output shouldn't be completely different from input
    # (unless weights are weird)
    correlation = np.corrcoef(x.flatten(), output.flatten())[0, 1]
    
    # Should have some correlation due to residuals
    assert correlation > 0.1, f"Residual connection may not be working (correlation: {correlation:.3f})"
    
    print(f"[PASS] Residual connection test passed! (correlation: {correlation:.3f})")


def test_stacking():
    """Test that multiple encoder blocks can be stacked."""
    n_layers = 3
    d_model = 64
    
    encoders = [EncoderBlock(d_model, n_heads=4, d_ff=256) for _ in range(n_layers)]
    
    x = np.random.randn(2, 8, d_model)
    
    for encoder in encoders:
        x = encoder.forward(x)
    
    assert x.shape == (2, 8, d_model), f"Final shape: {x.shape}"
    
    print(f"[PASS] Stacking test passed! ({n_layers} layers)")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("TESTING ENCODER BLOCK")
    print("=" * 50)
    
    try:
        test_layer_norm()
        test_feed_forward()
        test_encoder_shape()
        test_residual_connection()
        test_stacking()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        
    except NotImplementedError as e:
        print(f"\n[TODO] {e}")
        print("Implement the forward method and run again!")
        
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        print("Check your implementation!")


if __name__ == "__main__":
    run_all_tests()
