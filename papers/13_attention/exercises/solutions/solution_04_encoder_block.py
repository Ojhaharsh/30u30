"""
Solution 4: Encoder Block

Complete implementation with all components.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def multi_head_attention(query, key, value, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """Multi-head attention."""
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
    
    Unlike BatchNorm:
    - Normalizes across features (not batch)
    - Works on single samples
    - Is position-independent
    
    This is crucial for Transformers where batch sizes can be 1 at inference.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    Structure: Linear -> ReLU -> Linear
    
    Key insight: Same transformation applied to each position independently.
    This is where the model does "thinking" after gathering information via attention.
    
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
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class EncoderBlock:
    """
    Transformer Encoder Block.
    
    Structure (Post-Norm as in original paper):
        1. Self-Attention
        2. Residual + LayerNorm
        3. Feed-Forward
        4. Residual + LayerNorm
    
    Residual connections are ESSENTIAL:
    - Allow gradients to flow directly
    - Enable training of very deep models
    - Each layer learns a "refinement" of the input
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
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
        """
        # Step 1: Self-attention
        attn_output = multi_head_attention(
            x, x, x, 
            self.W_Q, self.W_K, self.W_V, self.W_O,
            self.n_heads, mask
        )
        
        # Step 2: Residual + Layer Norm
        x = self.norm1.forward(x + attn_output)
        
        # Step 3: Feed-forward
        ff_output = self.ff.forward(x)
        
        # Step 4: Residual + Layer Norm
        x = self.norm2.forward(x + ff_output)
        
        return x


# =============================================================================
# TESTS
# =============================================================================

def test_layer_norm():
    """Test layer normalization."""
    ln = LayerNorm(64)
    x = np.random.randn(2, 10, 64) * 5 + 3
    
    output = ln.forward(x)
    
    mean = output.mean(axis=-1)
    std = output.std(axis=-1)
    
    assert np.allclose(mean, 0, atol=0.1), f"Mean should be ~0"
    assert np.allclose(std, 1, atol=0.1), f"Std should be ~1"
    
    print("[PASS] Layer norm test passed!")


def test_feed_forward():
    """Test feed-forward network."""
    ff = FeedForward(64, 256)
    x = np.random.randn(2, 10, 64)
    
    output = ff.forward(x)
    
    assert output.shape == x.shape
    
    print("[PASS] Feed-forward test passed!")


def test_encoder_shape():
    """Test encoder block output shape."""
    encoder = EncoderBlock(d_model=64, n_heads=4, d_ff=256)
    x = np.random.randn(2, 10, 64)
    
    output = encoder.forward(x)
    
    assert output.shape == x.shape
    
    print("[PASS] Encoder shape test passed!")


def test_residual_connection():
    """Test that residual connections are working."""
    np.random.seed(42)
    
    encoder = EncoderBlock(d_model=64, n_heads=4, d_ff=256)
    x = np.random.randn(1, 3, 64)
    
    output = encoder.forward(x)
    
    correlation = np.corrcoef(x.flatten(), output.flatten())[0, 1]
    
    assert correlation > 0.1, "Residual connection may not be working"
    
    print(f"[PASS] Residual connection test passed! (correlation: {correlation:.3f})")


def test_stacking():
    """Test that multiple encoder blocks can be stacked."""
    n_layers = 3
    d_model = 64
    
    encoders = [EncoderBlock(d_model, n_heads=4, d_ff=256) for _ in range(n_layers)]
    
    x = np.random.randn(2, 8, d_model)
    
    for encoder in encoders:
        x = encoder.forward(x)
    
    assert x.shape == (2, 8, d_model)
    
    print(f"[PASS] Stacking test passed! ({n_layers} layers)")


def demonstrate_encoder():
    """Demonstrate encoder block."""
    print("\n" + "=" * 50)
    print("ENCODER BLOCK DEMONSTRATION")
    print("=" * 50)
    
    np.random.seed(42)
    
    d_model, n_heads, d_ff = 64, 4, 256
    
    encoder = EncoderBlock(d_model, n_heads, d_ff)
    
    # Input: 5 tokens
    x = np.random.randn(1, 5, d_model)
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  d_k (per head): {d_model // n_heads}")
    
    output = encoder.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Show that output is related to input (residual)
    correlation = np.corrcoef(x.flatten(), output.flatten())[0, 1]
    print(f"\nInput-output correlation: {correlation:.3f}")
    print("(Residual connections maintain some input signal)")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("SOLUTION 4: ENCODER BLOCK")
    print("=" * 50)
    
    test_layer_norm()
    test_feed_forward()
    test_encoder_shape()
    test_residual_connection()
    test_stacking()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    
    demonstrate_encoder()


if __name__ == "__main__":
    run_all_tests()
