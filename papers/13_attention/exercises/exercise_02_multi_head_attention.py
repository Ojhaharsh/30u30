"""
Exercise 2: Multi-Head Attention

Implement multi-head attention - the key innovation that allows
the Transformer to attend to information from different
representation subspaces.

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Reference attention implementation."""
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V), weights


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Instead of one attention function, we project Q, K, V into h different
    subspaces and apply attention in parallel. This allows the model to
    jointly attend to information from different representation subspaces.
    
    Key insight: Different heads learn different relationships!
    - One head might learn syntactic relationships
    - Another might learn semantic relationships
    - Another might learn positional patterns
    
    TODO: Implement the forward method.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Initialize projection matrices
        # Xavier initialization for stable training
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        """
        Compute multi-head attention.
        
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional (batch, 1, seq_q, seq_k)
            
        Returns:
            output: (batch, seq_q, d_model)
        
        Steps:
            1. Project Q, K, V using learned weight matrices
            2. Reshape to split heads: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
            3. Apply scaled dot-product attention to each head
            4. Concatenate heads: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
            5. Final linear projection
        """
        # ==========================================
        # TODO: Implement multi-head attention
        # ==========================================
        
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]
        
        # Step 1: Linear projections
        # Q = query @ self.W_Q  # (batch, seq_q, d_model)
        # K = ???
        # V = ???
        
        # Step 2: Reshape for multi-head
        # From (batch, seq, d_model) to (batch, n_heads, seq, d_k)
        # Hint: reshape to (batch, seq, n_heads, d_k), then transpose axes
        # Q = Q.reshape(batch_size, seq_q, self.n_heads, self.d_k).transpose(???)
        # K = ???
        # V = ???
        
        # Step 3: Apply attention
        # Use the provided scaled_dot_product_attention function
        # attn_output, self.attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        # From (batch, n_heads, seq_q, d_k) to (batch, seq_q, d_model)
        # attn_output = attn_output.transpose(???).reshape(???)
        
        # Step 5: Final projection
        # output = attn_output @ self.W_O
        
        # return output
        
        raise NotImplementedError("Implement multi-head attention!")


# =============================================================================
# TESTS
# =============================================================================

def test_mha_shape():
    """Test output shape."""
    batch, seq_len, d_model, n_heads = 2, 10, 64, 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = np.random.randn(batch, seq_len, d_model)
    
    output = mha.forward(x, x, x)
    
    expected_shape = (batch, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    
    print("[PASS] Shape test passed!")


def test_mha_different_lengths():
    """Test with different sequence lengths for Q and K/V."""
    batch, seq_q, seq_k, d_model, n_heads = 2, 6, 10, 64, 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    query = np.random.randn(batch, seq_q, d_model)
    key = np.random.randn(batch, seq_k, d_model)
    value = np.random.randn(batch, seq_k, d_model)
    
    output = mha.forward(query, key, value)
    
    expected_shape = (batch, seq_q, d_model)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    
    print("[PASS] Different lengths test passed!")


def test_attention_weights_shape():
    """Test that attention weights have correct shape."""
    batch, seq_len, d_model, n_heads = 2, 8, 64, 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = np.random.randn(batch, seq_len, d_model)
    
    _ = mha.forward(x, x, x)
    
    expected_shape = (batch, n_heads, seq_len, seq_len)
    assert mha.attention_weights.shape == expected_shape, \
        f"Expected {expected_shape}, got {mha.attention_weights.shape}"
    
    print("[PASS] Attention weights shape test passed!")


def test_different_heads():
    """Verify different heads produce different attention patterns."""
    np.random.seed(42)
    
    batch, seq_len, d_model, n_heads = 1, 5, 64, 4
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = np.random.randn(batch, seq_len, d_model)
    
    _ = mha.forward(x, x, x)
    
    # Different heads should have different attention patterns
    # (unless by extreme coincidence)
    head_patterns = [mha.attention_weights[0, h] for h in range(n_heads)]
    
    # Check that not all heads are identical
    all_same = True
    for i in range(1, n_heads):
        if not np.allclose(head_patterns[0], head_patterns[i], atol=0.1):
            all_same = False
            break
    
    assert not all_same, "All heads have identical patterns - check projection initialization"
    
    print("[PASS] Different heads test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("TESTING MULTI-HEAD ATTENTION")
    print("=" * 50)
    
    try:
        test_mha_shape()
        test_mha_different_lengths()
        test_attention_weights_shape()
        test_different_heads()
        
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
