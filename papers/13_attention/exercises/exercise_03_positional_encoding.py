"""
Exercise 3: Positional Encoding

Implement sinusoidal positional encoding from the Transformer paper.

Without recurrence, the model has no sense of position!
"cat sat mat" = "mat cat sat" (same to pure attention)

Solution: Add position information using sine/cosine waves:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import numpy as np


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding.
    
    Key insights:
    1. Each position gets a unique encoding
    2. Relative positions can be computed as linear functions
    3. Can extrapolate to longer sequences than seen in training
    
    The encoding uses sine waves of different frequencies:
    - Low dimensions: Long wavelengths (global position)
    - High dimensions: Short wavelengths (local details)
    
    TODO: Implement the _create_encoding method.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length to precompute
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute positional encodings
        self.pe = self._create_encoding(max_len, d_model)
    
    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """
        Create the sinusoidal encoding matrix.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            pe: Positional encoding matrix of shape (max_len, d_model)
        
        Formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Steps:
            1. Create position indices: [0, 1, 2, ..., max_len-1]
            2. Create dimension indices: [0, 2, 4, ..., d_model-2]
            3. Compute the division term: 10000^(2i/d_model)
               Hint: Use exp(log) for numerical stability
            4. Apply sin to even indices, cos to odd indices
        """
        # ==========================================
        # TODO: Implement positional encoding
        # ==========================================
        
        # Step 1: Initialize output matrix
        # pe = np.zeros((max_len, d_model))
        
        # Step 2: Create position indices [0, 1, 2, ..., max_len-1]
        # Shape should be (max_len, 1) for broadcasting
        # position = np.arange(max_len)[:, np.newaxis]
        
        # Step 3: Compute division term
        # div_term = 10000^(2i/d_model) for i in [0, 1, ..., d_model/2 - 1]
        # Hint: 10000^x = exp(x * log(10000))
        # div_term = np.exp(np.arange(0, d_model, 2) * ???)
        
        # Step 4: Apply sin to even indices (0, 2, 4, ...)
        # pe[:, 0::2] = np.sin(???)
        
        # Step 5: Apply cos to odd indices (1, 3, 5, ...)
        # pe[:, 1::2] = np.cos(???)
        
        # return pe
        
        raise NotImplementedError("Implement positional encoding!")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            
        Returns:
            Embeddings + positional encoding
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


# =============================================================================
# TESTS
# =============================================================================

def test_pe_shape():
    """Test output shape."""
    d_model, max_len = 64, 100
    
    pe = PositionalEncoding(d_model, max_len)
    
    assert pe.pe.shape == (max_len, d_model), \
        f"Expected ({max_len}, {d_model}), got {pe.pe.shape}"
    
    print("[PASS] Shape test passed!")


def test_pe_range():
    """Test that values are in [-1, 1]."""
    pe = PositionalEncoding(128, 1000)
    
    assert np.all(pe.pe >= -1) and np.all(pe.pe <= 1), \
        "PE values should be in [-1, 1]"
    
    print("[PASS] Range test passed!")


def test_unique_positions():
    """Test that each position has a unique encoding."""
    pe = PositionalEncoding(64, 100)
    
    # Check that no two positions have identical encodings
    for i in range(10):
        for j in range(i + 1, 10):
            if np.allclose(pe.pe[i], pe.pe[j], atol=1e-6):
                raise AssertionError(f"Positions {i} and {j} have identical encodings!")
    
    print("[PASS] Unique positions test passed!")


def test_nearby_similarity():
    """Test that nearby positions have similar encodings."""
    pe = PositionalEncoding(64, 100)
    
    # Nearby positions should be more similar than distant ones
    dist_1 = np.linalg.norm(pe.pe[0] - pe.pe[1])
    dist_10 = np.linalg.norm(pe.pe[0] - pe.pe[10])
    dist_50 = np.linalg.norm(pe.pe[0] - pe.pe[50])
    
    assert dist_1 < dist_10 < dist_50, \
        f"Nearby positions should be more similar: dist_1={dist_1:.2f}, dist_10={dist_10:.2f}, dist_50={dist_50:.2f}"
    
    print(f"[PASS] Similarity test passed! (dist_1={dist_1:.2f} < dist_10={dist_10:.2f} < dist_50={dist_50:.2f})")


def test_sinusoidal_pattern():
    """Test that the encoding follows sinusoidal pattern."""
    pe = PositionalEncoding(64, 100)
    
    # First dimension (2i=0) should be sin(pos / 10000^0) = sin(pos)
    expected_dim0 = np.sin(np.arange(100))
    
    # Allow some tolerance due to the division term
    # The first dimension should be a sine wave
    dim0 = pe.pe[:, 0]
    
    # Check it's actually sinusoidal (multiple full cycles would be present)
    # by checking it oscillates
    sign_changes = np.sum(np.diff(np.sign(dim0)) != 0)
    
    assert sign_changes > 10, \
        f"First dimension should oscillate (got {sign_changes} sign changes)"
    
    print(f"[PASS] Sinusoidal pattern test passed! ({sign_changes} oscillations)")


def test_forward_addition():
    """Test that forward properly adds PE to input."""
    pe = PositionalEncoding(32, 100)
    
    x = np.ones((2, 10, 32)) * 5
    output = pe.forward(x)
    
    # Output should be input + positional encoding
    expected = x + pe.pe[:10]
    
    assert np.allclose(output, expected), \
        "Forward should add PE to input"
    
    print("[PASS] Forward addition test passed!")


def visualize_pe():
    """Visualize positional encoding (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt
        
        pe = PositionalEncoding(128, 100)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap
        ax = axes[0]
        im = ax.imshow(pe.pe.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('Positional Encoding Heatmap')
        plt.colorbar(im, ax=ax)
        
        # First few dimensions
        ax = axes[1]
        for dim in range(0, 8, 2):
            ax.plot(pe.pe[:, dim], label=f'dim {dim}', alpha=0.8)
        ax.set_xlabel('Position')
        ax.set_ylabel('Encoding Value')
        ax.set_title('Positional Encoding Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('positional_encoding.png', dpi=150)
        plt.show()
        
        print("[VIZ] Saved visualization to 'positional_encoding.png'")
        
    except ImportError:
        print("[VIZ] Matplotlib not available - skipping visualization")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("TESTING POSITIONAL ENCODING")
    print("=" * 50)
    
    try:
        test_pe_shape()
        test_pe_range()
        test_unique_positions()
        test_nearby_similarity()
        test_sinusoidal_pattern()
        test_forward_addition()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        
        # Try visualization
        print("\nGenerating visualization...")
        visualize_pe()
        
    except NotImplementedError as e:
        print(f"\n[TODO] {e}")
        print("Implement the _create_encoding method and run again!")
        
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        print("Check your implementation!")


if __name__ == "__main__":
    run_all_tests()
