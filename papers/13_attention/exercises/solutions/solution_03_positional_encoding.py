"""
Solution 3: Positional Encoding

Complete implementation with visualization.
"""

import numpy as np


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding.
    
    Why sinusoids?
    1. Unique encoding for every position
    2. Distances can be computed linearly (PE[pos+k] = f(PE[pos], k))
    3. Can extrapolate to longer sequences than training
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    The 10000 base creates waves from very slow (dim 0) to fast (dim d_model-1).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._create_encoding(max_len, d_model)
    
    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create the sinusoidal encoding matrix."""
        # Initialize output
        pe = np.zeros((max_len, d_model))
        
        # Position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1) for broadcasting
        position = np.arange(max_len)[:, np.newaxis]
        
        # Division term: 10000^(2i/d_model) for i in [0, 1, ..., d_model/2 - 1]
        # Using exp(log) for numerical stability:
        # 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        # We compute the INVERSE for division: exp(-2i * log(10000) / d_model)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        # Shape: (d_model // 2,)
        
        # Apply sin to even indices (0, 2, 4, ...)
        # position * div_term broadcasts: (max_len, 1) * (d_model//2,) -> (max_len, d_model//2)
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input embeddings."""
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
    
    for i in range(10):
        for j in range(i + 1, 10):
            if np.allclose(pe.pe[i], pe.pe[j], atol=1e-6):
                raise AssertionError(f"Positions {i} and {j} have identical encodings!")
    
    print("[PASS] Unique positions test passed!")


def test_nearby_similarity():
    """Test that nearby positions have similar encodings."""
    pe = PositionalEncoding(64, 100)
    
    dist_1 = np.linalg.norm(pe.pe[0] - pe.pe[1])
    dist_10 = np.linalg.norm(pe.pe[0] - pe.pe[10])
    dist_50 = np.linalg.norm(pe.pe[0] - pe.pe[50])
    
    assert dist_1 < dist_10 < dist_50, \
        f"Nearby positions should be more similar: {dist_1:.2f} < {dist_10:.2f} < {dist_50:.2f}"
    
    print(f"[PASS] Similarity test passed!")


def test_sinusoidal_pattern():
    """Test that the encoding follows sinusoidal pattern."""
    pe = PositionalEncoding(64, 100)
    
    dim0 = pe.pe[:, 0]
    sign_changes = np.sum(np.diff(np.sign(dim0)) != 0)
    
    assert sign_changes > 10, f"First dimension should oscillate"
    
    print(f"[PASS] Sinusoidal pattern test passed!")


def test_forward_addition():
    """Test that forward properly adds PE to input."""
    pe = PositionalEncoding(32, 100)
    
    x = np.ones((2, 10, 32)) * 5
    output = pe.forward(x)
    
    expected = x + pe.pe[:10]
    
    assert np.allclose(output, expected), "Forward should add PE to input"
    
    print("[PASS] Forward addition test passed!")


def visualize_pe():
    """Create ASCII visualization of positional encoding."""
    print("\n" + "=" * 50)
    print("POSITIONAL ENCODING VISUALIZATION")
    print("=" * 50)
    
    pe = PositionalEncoding(8, 20)
    
    print("\nFirst 10 positions, first 8 dimensions:")
    print("(Values scaled to characters: ' ' < '.' < 'o' < '@')")
    print()
    
    print("Pos  | Dim 0 1 2 3 4 5 6 7")
    print("-" * 30)
    
    for pos in range(10):
        chars = []
        for dim in range(8):
            val = pe.pe[pos, dim]
            if val < -0.5:
                char = ' '
            elif val < 0:
                char = '.'
            elif val < 0.5:
                char = 'o'
            else:
                char = '@'
            chars.append(char)
        print(f"  {pos:2d} |     {' '.join(chars)}")
    
    print("\nNotice: Different frequencies for different dimensions!")
    print("Low dims (0-1): Slow changing (long wavelength)")
    print("High dims (6-7): Fast changing (short wavelength)")


def demonstrate_relative_positions():
    """Show that relative positions can be computed."""
    print("\n" + "=" * 50)
    print("RELATIVE POSITION PROPERTY")
    print("=" * 50)
    
    pe = PositionalEncoding(64, 100)
    
    # For any fixed offset k, PE(pos+k) can be expressed as a
    # linear transformation of PE(pos)
    print("\nDot product similarity between positions:")
    print("(Higher = more similar)")
    print()
    
    ref_pos = 0
    for offset in [1, 5, 10, 20, 50]:
        similarity = np.dot(pe.pe[ref_pos], pe.pe[ref_pos + offset])
        print(f"  pos 0 vs pos {offset:2d}: similarity = {similarity:.3f}")
    
    print("\nNote: Similarity decreases with distance!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("SOLUTION 3: POSITIONAL ENCODING")
    print("=" * 50)
    
    test_pe_shape()
    test_pe_range()
    test_unique_positions()
    test_nearby_similarity()
    test_sinusoidal_pattern()
    test_forward_addition()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    
    visualize_pe()
    demonstrate_relative_positions()


if __name__ == "__main__":
    run_all_tests()
