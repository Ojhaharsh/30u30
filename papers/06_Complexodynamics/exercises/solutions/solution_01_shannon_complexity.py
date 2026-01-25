"""
Solution to Exercise 1: Shannon Complexity

Complete implementation with detailed comments.
"""

import numpy as np
from collections import Counter
from typing import Dict


def shannon_complexity(sequence: str) -> float:
    """
    Calculate Shannon entropy of a sequence.
    
    Formula: H = -Σ p_i * log2(p_i)
    where p_i is the probability of symbol i.
    """
    # Edge case: empty sequence
    if len(sequence) == 0:
        return 0.0
    
    # Count frequency of each symbol
    counts = Counter(sequence)
    total = len(sequence)
    
    # Calculate probabilities
    probabilities = [count / total for count in counts.values()]
    
    # Apply Shannon entropy formula
    # Note: We filter out p=0 to avoid log(0)
    entropy = 0.0
    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy -= p * np.log2(p)
    
    return entropy


def complexity_bits_per_base(sequence: str, alphabet_size: int = 4) -> float:
    """
    Calculate complexity as percentage of maximum possible.
    
    Maximum entropy for N symbols is log2(N).
    For DNA (4 bases), max is log2(4) = 2 bits.
    """
    if len(sequence) == 0:
        return 0.0
    
    C = shannon_complexity(sequence)
    C_max = np.log2(alphabet_size)
    
    return C / C_max


def local_complexity_profile(sequence: str, window_size: int = 100) -> Dict:
    """
    Calculate sliding window complexity across sequence.
    
    Returns:
        - positions: Start position of each window
        - complexities: Complexity value at each position
        - mean: Average complexity
        - std: Standard deviation of complexity
    """
    if len(sequence) < window_size:
        # Sequence too short for window
        return {
            'positions': [0],
            'complexities': [shannon_complexity(sequence)],
            'mean': shannon_complexity(sequence),
            'std': 0.0
        }
    
    positions = []
    complexities = []
    
    # Slide window across sequence
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        C = shannon_complexity(window)
        
        positions.append(i)
        complexities.append(C)
    
    # Convert to numpy arrays for statistics
    complexities_array = np.array(complexities)
    
    return {
        'positions': positions,
        'complexities': complexities,
        'mean': np.mean(complexities_array),
        'std': np.std(complexities_array)
    }


def compare_sequences(sequences: Dict[str, str]) -> None:
    """
    Compare complexity of multiple sequences.
    
    Prints a formatted table showing:
    - Sequence name
    - Length
    - Complexity (bits/symbol)
    - Percentage of maximum
    """
    print("\n" + "="*80)
    print("SEQUENCE COMPLEXITY COMPARISON")
    print("="*80)
    print(f"{'Name':<25} {'Length':<10} {'Complexity':<15} {'% of Max':<15}")
    print("-"*80)
    
    for name, seq in sequences.items():
        length = len(seq)
        C = shannon_complexity(seq)
        pct = complexity_bits_per_base(seq, alphabet_size=4) * 100
        
        print(f"{name:<25} {length:<10} {C:<15.4f} {pct:<15.1f}%")
    
    print("="*80)


# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def complexity_from_counts(counts: Dict[str, int]) -> float:
    """
    Calculate complexity directly from symbol counts.
    
    Useful when you already have frequency data.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    return entropy


def normalized_entropy(sequence: str, alphabet_size: int = 4) -> float:
    """
    Entropy normalized to [0, 1] range.
    
    0 = minimum complexity (uniform)
    1 = maximum complexity (all symbols equal frequency)
    """
    if len(sequence) == 0 or alphabet_size == 1:
        return 0.0
    
    C = shannon_complexity(sequence)
    C_max = np.log2(alphabet_size)
    
    return C / C_max


def information_content(sequence: str) -> float:
    """
    Total information content (bits) in entire sequence.
    
    This is entropy * length.
    Represents total "surprise" in the sequence.
    """
    return shannon_complexity(sequence) * len(sequence)


# ============================================================================
# TEST CASES (Same as exercise)
# ============================================================================

def run_tests():
    """Run all test cases."""
    print("="*70)
    print("TESTING SOLUTION 1: Shannon Complexity")
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
    bacteria = "GCGC" * 250
    C_bacteria = shannon_complexity(bacteria)
    assert 1.9 < C_bacteria < 2.0, f"Expected ~2.0, got {C_bacteria}"
    
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


def demo():
    """Demonstrate the solution."""
    print("\n" + "="*70)
    print("DEMO: Shannon Complexity Solution")
    print("="*70)
    
    sequences = {
        'Uniform': 'AAAA',
        'Binary': 'AACC',
        'Ternary': 'AACCGG',
        'Quaternary (Max)': 'AACCGGTT',
        'Skewed': 'AAAAACGT',
    }
    
    compare_sequences(sequences)
    
    # Show local complexity profile
    print("\n" + "="*70)
    print("LOCAL COMPLEXITY PROFILE")
    print("="*70)
    
    # Create test sequence with regions of different complexity
    low = "A" * 300
    medium = "AACC" * 75
    high = "ACGT" * 75
    test_seq = low + medium + high
    
    profile = local_complexity_profile(test_seq, window_size=100)
    
    print(f"Sequence length: {len(test_seq)}")
    print(f"Number of windows: {len(profile['positions'])}")
    print(f"Mean complexity: {profile['mean']:.4f} bits/base")
    print(f"Std deviation: {profile['std']:.4f} bits/base")
    print(f"Min complexity: {min(profile['complexities']):.4f} bits/base")
    print(f"Max complexity: {max(profile['complexities']):.4f} bits/base")
    

if __name__ == '__main__':
    run_tests()
    demo()
