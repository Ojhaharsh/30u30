"""
Exercise 1: Scaled Dot-Product Attention

Implement the core attention mechanism from "Attention Is All You Need".

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

This is the foundation of the entire Transformer architecture!
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
    
    The key insight: attention is a learned, differentiable lookup!
    
    Q (Query): What am I looking for?
    K (Key): What do I contain? 
    V (Value): What do I return if matched?
    
    TODO: Implement the forward method.
    """
    
    def __init__(self):
        self.attention_weights = None  # Cache for inspection
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries of shape (batch, n_heads, seq_q, d_k)
            K: Keys of shape (batch, n_heads, seq_k, d_k)
            V: Values of shape (batch, n_heads, seq_k, d_v)
            mask: Optional boolean mask (batch, 1, seq_q, seq_k)
                  True values are masked (set to -inf before softmax)
        
        Returns:
            output: (batch, n_heads, seq_q, d_v)
        
        Steps:
            1. Compute attention scores: Q @ K^T
            2. Scale by sqrt(d_k) to prevent gradient vanishing
            3. Apply mask if provided
            4. Softmax to get attention weights
            5. Multiply by V to get output
        """
        # ==========================================
        # TODO: Implement scaled dot-product attention
        # ==========================================
        
        # Step 1: Get the dimension for scaling
        # d_k = K.shape[-1]
        
        # Step 2: Compute raw attention scores (Q @ K^T)
        # Remember: K needs to be transposed on last two dims
        # scores = ???
        
        # Step 3: Scale by sqrt(d_k)
        # scores = scores / ???
        
        # Step 4: Apply mask if provided
        # Masked positions should become -infinity (or very large negative)
        # if mask is not None:
        #     scores = ???
        
        # Step 5: Apply softmax to get attention weights
        # self.attention_weights = softmax(???)
        
        # Step 6: Multiply by values
        # output = ???
        
        # return output
        
        raise NotImplementedError("Implement scaled dot-product attention!")


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
    
    # Weights should sum to 1 along the key dimension
    weight_sums = attention.attention_weights.sum(axis=-1)
    
    assert np.allclose(weight_sums, 1.0, atol=1e-5), \
        f"Attention weights should sum to 1, got {weight_sums}"
    
    print("[PASS] Attention weights sum test passed!")


def test_masking():
    """Test that masking works correctly."""
    Q = np.ones((1, 1, 3, 4))
    K = np.ones((1, 1, 3, 4))
    V = np.array([[[[1, 0], [2, 0], [3, 0]]]])  # Shape: (1, 1, 3, 2)
    
    # Mask out position 2 (the "3")
    mask = np.zeros((1, 1, 3, 3), dtype=bool)
    mask[0, 0, :, 2] = True  # Mask key position 2
    
    attention = ScaledDotProductAttention()
    output = attention.forward(Q, K, V, mask=mask)
    
    # With position 2 masked, output should be average of positions 0 and 1
    # Values are [1, 0] and [2, 0], so average is [1.5, 0]
    # (assuming equal attention without the masked position)
    
    # Just verify the masked position gets near-zero attention
    weights = attention.attention_weights
    assert weights[0, 0, 0, 2] < 0.01, \
        f"Masked position should have near-zero weight, got {weights[0, 0, 0, 2]}"
    
    print("[PASS] Masking test passed!")


def test_scaling_effect():
    """Test that scaling prevents extreme softmax values."""
    np.random.seed(42)
    
    # Large dimension - without scaling, dot products would be huge
    d_k = 512
    Q = np.random.randn(1, 1, 1, d_k)
    K = np.random.randn(1, 1, 10, d_k)
    V = np.random.randn(1, 1, 10, d_k)
    
    attention = ScaledDotProductAttention()
    _ = attention.forward(Q, K, V)
    
    weights = attention.attention_weights[0, 0, 0]
    
    # With proper scaling, no single weight should be too close to 1
    # (unless there's a true match)
    max_weight = weights.max()
    
    # In random data with scaling, max weight should be reasonable
    assert max_weight < 0.9, \
        f"Scaling might be off - max weight is {max_weight}"
    
    print(f"[PASS] Scaling test passed! (max weight: {max_weight:.3f})")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("TESTING SCALED DOT-PRODUCT ATTENTION")
    print("=" * 50)
    
    try:
        test_attention_shape()
        test_attention_weights_sum()
        test_masking()
        test_scaling_effect()
        
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
