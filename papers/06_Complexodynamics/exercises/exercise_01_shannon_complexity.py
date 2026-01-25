"""
Exercise 1: Shannon Complexity (⭐ Easy - 30 minutes)

Implement Shannon entropy calculator for DNA sequences.

Learning goals:
- Understand information content measurement
- Frequency counting and probability calculation
- Edge case handling

Key formula:
    C = -Σ p_i * log2(p_i)

where p_i is the probability of symbol i.
"""

import numpy as np
from collections import Counter
from typing import Dict


def shannon_complexity(sequence: str) -> float:
    """
    Calculate Shannon entropy of a sequence.
    
    Args:
        sequence: String of symbols (e.g., "ACGTACGT" for DNA)
        
    Returns:
        Shannon entropy in bits per symbol
        
    Examples:
        >>> shannon_complexity("AAAA")  # All same
        0.0
        >>> shannon_complexity("ACGT")  # Maximum diversity
        2.0
        >>> shannon_complexity("AACCGGTT")  # Still maximum
        2.0
        >>> shannon_complexity("AAAACCCC")  # Half-half
        1.0
    
    TODO: Implement this function!
    
    Hints:
    1. Handle empty sequence edge case
    2. Count frequency of each symbol
    3. Convert counts to probabilities
    4. Apply Shannon entropy formula
    5. Use np.log2 for base-2 logarithm
    """
    # TODO: Your code here
    pass


def complexity_bits_per_base(sequence: str, alphabet_size: int = 4) -> float:
    """
    Calculate complexity as percentage of maximum possible.
    
    Args:
        sequence: DNA/protein sequence
        alphabet_size: Number of possible symbols (4 for DNA, 20 for protein)
        
    Returns:
        Complexity as fraction of maximum (0 to 1)
        
    Example:
        >>> complexity_bits_per_base("ACGT", alphabet_size=4)
        1.0  # Maximum complexity
        >>> complexity_bits_per_base("AAAA", alphabet_size=4)
        0.0  # Minimum complexity
    
    TODO: Implement this function!
    
    Hint: Maximum entropy for N symbols is log2(N)
    """
    # TODO: Your code here
    pass


def local_complexity_profile(sequence: str, window_size: int = 100) -> Dict:
    """
    Calculate sliding window complexity across sequence.
    
    Useful for finding low/high complexity regions in genomes.
    
    Args:
        sequence: Full sequence
        window_size: Size of sliding window
        
    Returns:
        Dictionary with:
            - 'positions': window start positions
            - 'complexities': complexity at each position
            - 'mean': average complexity
            - 'std': standard deviation
    
    Example:
        >>> seq = "A"*100 + "ACGT"*25  # Low then high complexity
        >>> profile = local_complexity_profile(seq, window_size=50)
        >>> profile['complexities'][0] < profile['complexities'][-1]
        True
    
    TODO: Implement this function!
    
    Hints:
    1. Slide window across sequence
    2. Calculate complexity for each window
    3. Store results and compute statistics
    """
    # TODO: Your code here
    pass


def compare_sequences(sequences: Dict[str, str]) -> None:
    """
    Compare complexity of multiple sequences.
    
    Args:
        sequences: Dict mapping names to sequences
        
    Prints comparison table.
    
    TODO: Implement this function!
    """
    # TODO: Your code here
    pass


# ============================================================================
# TEST CASES
# ============================================================================

def run_tests():
    """Run all test cases."""
    print("="*70)
    print("TESTING EXERCISE 1: Shannon Complexity")
    print("="*70)
    
    # Test 1: Edge cases
    print("\n[Test 1] Edge cases...")
    assert shannon_complexity("") == 0.0, "Empty sequence should have 0 complexity"
    assert shannon_complexity("A") == 0.0, "Single symbol should have 0 complexity"
    print("✅ Edge cases passed")
    
    # Test 2: Uniform sequences
    print("\n[Test 2] Uniform sequences...")
    assert shannon_complexity("AAAA") == 0.0, "All same should be 0"
    assert shannon_complexity("GGGGGGGG") == 0.0, "All same should be 0"
    print("✅ Uniform sequences passed")
    
    # Test 3: Maximum diversity
    print("\n[Test 3] Maximum diversity...")
    C_max = shannon_complexity("ACGT")
    assert abs(C_max - 2.0) < 0.01, f"Expected 2.0, got {C_max}"
    
    C_balanced = shannon_complexity("AACCGGTT")
    assert abs(C_balanced - 2.0) < 0.01, f"Expected 2.0, got {C_balanced}"
    print("✅ Maximum diversity passed")
    
    # Test 4: Intermediate complexity
    print("\n[Test 4] Intermediate complexity...")
    C_half = shannon_complexity("AAAACCCC")
    assert abs(C_half - 1.0) < 0.01, f"Expected 1.0, got {C_half}"
    
    C_skewed = shannon_complexity("AAAC")
    expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert abs(C_skewed - expected) < 0.01, f"Expected {expected}, got {C_skewed}"
    print("✅ Intermediate complexity passed")
    
    # Test 5: Real-ish sequences
    print("\n[Test 5] Realistic sequences...")
    # Bacterial genome (GC-rich)
    bacteria = "GCGC" * 250
    C_bacteria = shannon_complexity(bacteria)
    assert 1.9 < C_bacteria < 2.0, f"Expected ~2.0, got {C_bacteria}"
    
    # AT-rich region
    at_rich = "AATT" * 250
    C_at = shannon_complexity(at_rich)
    assert 1.9 < C_at < 2.0, f"Expected ~2.0, got {C_at}"
    print("✅ Realistic sequences passed")
    
    # Test 6: Complexity percentage
    print("\n[Test 6] Complexity percentage...")
    assert complexity_bits_per_base("ACGT", 4) == 1.0
    assert complexity_bits_per_base("AAAA", 4) == 0.0
    print("✅ Complexity percentage passed")
    
    # Test 7: Local complexity
    print("\n[Test 7] Local complexity profile...")
    # Create sequence with varying complexity
    low_complex = "A" * 500
    high_complex = "ACGT" * 125
    test_seq = low_complex + high_complex
    
    profile = local_complexity_profile(test_seq, window_size=100)
    
    assert len(profile['positions']) > 0, "Should have positions"
    assert len(profile['complexities']) == len(profile['positions']), "Length mismatch"
    assert profile['complexities'][0] < profile['complexities'][-1], \
        "First window should be less complex than last"
    print("✅ Local complexity passed")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nBonus challenge: Analyze a real genome!")
    print("Download E. coli genome and calculate its complexity.")


def demo():
    """Demonstrate the functions."""
    print("\n" + "="*70)
    print("DEMO: Shannon Complexity in Action")
    print("="*70)
    
    # Example sequences
    sequences = {
        'Uniform (AAAA)': 'AAAA',
        'Binary (AACC)': 'AACC',
        'Max Diversity (ACGT)': 'ACGT',
        'Skewed (AAAACCCCGGGGTTTT)': 'AAAACCCCGGGGTTTT',
        'Random-ish': 'ACGTACGTACGTACGT',
    }
    
    print("\nSequence Complexity Comparison:")
    print("-" * 70)
    print(f"{'Sequence':<30} {'Complexity':<15} {'% of Max':<15}")
    print("-" * 70)
    
    for name, seq in sequences.items():
        C = shannon_complexity(seq)
        pct = complexity_bits_per_base(seq, alphabet_size=4) * 100
        print(f"{name:<30} {C:<15.4f} {pct:<15.1f}%")
    
    print("-" * 70)
    print("\n💡 Key insight: Maximum complexity (2.0 bits) = all bases equally likely!")
    

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        try:
            run_tests()
            demo()
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            print("\nHint: Check your implementation against the formulas in CHEATSHEET.md")
            sys.exit(1)
        except NotImplementedError:
            print("\n⚠️  Functions not yet implemented!")
            print("Remove 'pass' and add your code.")
            sys.exit(1)
