"""
Solution 2: Multi-Head Attention

Complete implementation with detailed explanations.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V), weights


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Key insights:
    1. Different heads capture different relationships
    2. Each head operates on a subset of dimensions (d_k = d_model / n_heads)
    3. Heads run in parallel, then concatenate
    
    The reshape is the trickiest part:
    - (batch, seq, d_model) -> (batch, seq, n_heads, d_k)
    - Then transpose to (batch, n_heads, seq, d_k) for attention
    """
    
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Initialize projection matrices
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
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
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]
        
        # Step 1: Linear projections
        # (batch, seq, d_model) @ (d_model, d_model) -> (batch, seq, d_model)
        Q = query @ self.W_Q
        K = key @ self.W_K
        V = value @ self.W_V
        
        # Step 2: Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        #
        # The reshape splits d_model into n_heads groups of d_k dimensions
        # The transpose puts heads as the second dimension for parallel processing
        Q = Q.reshape(batch_size, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Step 3: Apply attention to each head
        attn_output, self.attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, n_heads, seq_q, d_k)
        
        # Step 4: Concatenate heads
        # (batch, n_heads, seq_q, d_k) -> (batch, seq_q, n_heads, d_k) -> (batch, seq_q, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.d_model)
        
        # Step 5: Final linear projection
        output = attn_output @ self.W_O
        
        return output


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
    """Test with different sequence lengths."""
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
    """Verify different heads produce different patterns."""
    np.random.seed(42)
    
    batch, seq_len, d_model, n_heads = 1, 5, 64, 4
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = np.random.randn(batch, seq_len, d_model)
    
    _ = mha.forward(x, x, x)
    
    head_patterns = [mha.attention_weights[0, h] for h in range(n_heads)]
    
    all_same = True
    for i in range(1, n_heads):
        if not np.allclose(head_patterns[0], head_patterns[i], atol=0.1):
            all_same = False
            break
    
    assert not all_same, "All heads have identical patterns"
    
    print("[PASS] Different heads test passed!")


def demonstrate_multi_head():
    """Demonstrate multi-head attention."""
    print("\n" + "=" * 50)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Small example
    d_model, n_heads = 8, 2
    d_k = d_model // n_heads
    
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Input: 3 tokens, each with 8 features
    x = np.random.randn(1, 3, d_model)
    
    output = mha.forward(x, x, x)
    
    print(f"\nConfiguration: d_model={d_model}, n_heads={n_heads}, d_k={d_k}")
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print(f"\nAttention weights shape: {mha.attention_weights.shape}")
    print("(batch, n_heads, seq_q, seq_k)")
    
    print("\nHead 0 attention pattern:")
    print(mha.attention_weights[0, 0])
    
    print("\nHead 1 attention pattern:")
    print(mha.attention_weights[0, 1])
    
    print("\nNote: Each head learns a different attention pattern!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("SOLUTION 2: MULTI-HEAD ATTENTION")
    print("=" * 50)
    
    test_mha_shape()
    test_mha_different_lengths()
    test_attention_weights_shape()
    test_different_heads()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    
    demonstrate_multi_head()


if __name__ == "__main__":
    run_all_tests()
