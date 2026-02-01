"""
Solution 1: Scaled Dot-Product Attention

Complete implementation with explanations.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
    
    Why scale by sqrt(d_k)?
    -------------------------
    The dot product of two d_k-dimensional vectors has variance d_k
    (assuming inputs have unit variance). Large values push softmax
    into regions with tiny gradients, making learning difficult.
    
    Scaling by sqrt(d_k) keeps the variance at 1, ensuring softmax
    receives well-behaved inputs.
    """
    
    def __init__(self):
        self.attention_weights = None  # Cache for inspection
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries (batch, n_heads, seq_q, d_k)
            K: Keys (batch, n_heads, seq_k, d_k)
            V: Values (batch, n_heads, seq_k, d_v)
            mask: Optional boolean mask (batch, 1, seq_q, seq_k)
                  True values are masked (set to -inf before softmax)
        
        Returns:
            output: (batch, n_heads, seq_q, d_v)
        """
        # Step 1: Get the dimension for scaling
        d_k = K.shape[-1]
        
        # Step 2: Compute raw attention scores (Q @ K^T)
        # Q: (batch, heads, seq_q, d_k)
        # K: (batch, heads, seq_k, d_k) -> K^T: (batch, heads, d_k, seq_k)
        # Result: (batch, heads, seq_q, seq_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Step 3: Scale by sqrt(d_k)
        # This prevents the dot products from growing with d_k
        scores = scores / np.sqrt(d_k)
        
        # Step 4: Apply mask if provided
        # Masked positions should become -infinity (very large negative)
        # After softmax, they become essentially 0
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        # Step 5: Apply softmax to get attention weights
        # Each query's weights sum to 1
        self.attention_weights = softmax(scores, axis=-1)
        
        # Step 6: Multiply by values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, d_v)
        # -> (batch, heads, seq_q, d_v)
        output = np.matmul(self.attention_weights, V)
        
        return output


# =============================================================================
# TESTS
# =============================================================================

def test_attention_shape():
    """Test that output shape is correct."""
    batch, heads, seq_q, seq_k, d_k, d_v = 2, 4, 6, 8, 32, 32
    
    Q = np.random.randn(batch, heads, seq_q, d_k)
    K = np.random.randn(batch, heads, seq_k, d_k)
    V = np.random.randn(batch, heads, seq_k, d_v)
    
    attention = ScaledDotProductAttention()
    output = attention.forward(Q, K, V)
    
    expected_shape = (batch, heads, seq_q, d_v)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    
    print("[PASS] Shape test passed!")


def test_attention_weights_sum():
    """Test that attention weights sum to 1."""
    Q = np.random.randn(2, 4, 5, 32)
    K = np.random.randn(2, 4, 5, 32)
    V = np.random.randn(2, 4, 5, 32)
    
    attention = ScaledDotProductAttention()
    _ = attention.forward(Q, K, V)
    
    weight_sums = attention.attention_weights.sum(axis=-1)
    
    assert np.allclose(weight_sums, 1.0, atol=1e-5), \
        f"Attention weights should sum to 1, got {weight_sums}"
    
    print("[PASS] Attention weights sum test passed!")


def test_masking():
    """Test that masking works correctly."""
    Q = np.ones((1, 1, 3, 4))
    K = np.ones((1, 1, 3, 4))
    V = np.array([[[[1, 0], [2, 0], [3, 0]]]])
    
    mask = np.zeros((1, 1, 3, 3), dtype=bool)
    mask[0, 0, :, 2] = True  # Mask key position 2
    
    attention = ScaledDotProductAttention()
    output = attention.forward(Q, K, V, mask=mask)
    
    weights = attention.attention_weights
    assert weights[0, 0, 0, 2] < 0.01, \
        f"Masked position should have near-zero weight, got {weights[0, 0, 0, 2]}"
    
    print("[PASS] Masking test passed!")


def test_scaling_effect():
    """Test that scaling prevents extreme softmax values."""
    np.random.seed(42)
    
    d_k = 512
    Q = np.random.randn(1, 1, 1, d_k)
    K = np.random.randn(1, 1, 10, d_k)
    V = np.random.randn(1, 1, 10, d_k)
    
    attention = ScaledDotProductAttention()
    _ = attention.forward(Q, K, V)
    
    weights = attention.attention_weights[0, 0, 0]
    max_weight = weights.max()
    
    assert max_weight < 0.9, \
        f"Scaling might be off - max weight is {max_weight}"
    
    print(f"[PASS] Scaling test passed! (max weight: {max_weight:.3f})")


def demonstrate_attention():
    """Demonstrate how attention works."""
    print("\n" + "=" * 50)
    print("ATTENTION DEMONSTRATION")
    print("=" * 50)
    
    # Simple example: 3 words, 2 dimensions
    # Word 0 should attend to word 1 (similar keys)
    
    # Queries: what each position is looking for
    Q = np.array([[[[1, 0], [0, 1], [1, 1]]]])  # (1, 1, 3, 2)
    
    # Keys: what each position contains
    K = np.array([[[[1, 0], [1, 0], [0, 1]]]])  # Word 0 and 1 are similar
    
    # Values: what each position returns
    V = np.array([[[[10, 0], [20, 0], [30, 0]]]])
    
    attention = ScaledDotProductAttention()
    output = attention.forward(Q, K, V)
    
    print("\nQueries (what each position looks for):")
    print(Q[0, 0])
    
    print("\nKeys (what each position contains):")
    print(K[0, 0])
    
    print("\nValues (what each position returns):")
    print(V[0, 0])
    
    print("\nAttention weights:")
    print(attention.attention_weights[0, 0])
    print("(Each row shows how much each query attends to each key)")
    
    print("\nOutput (weighted sum of values):")
    print(output[0, 0])


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("SOLUTION 1: SCALED DOT-PRODUCT ATTENTION")
    print("=" * 50)
    
    test_attention_shape()
    test_attention_weights_sum()
    test_masking()
    test_scaling_effect()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    
    demonstrate_attention()


if __name__ == "__main__":
    run_all_tests()
