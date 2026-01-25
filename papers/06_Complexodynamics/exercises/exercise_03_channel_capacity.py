"""
Exercise 3: Channel Capacity (⭐⭐ Medium - 45 minutes)

Implement fidelity-complexity trade-off calculations.

Key formulas:
    Simple: C_max = -log2(μ * L)
    Gaussian: C_max = 0.5 * log2(1 / (2πe * σ²))
    Error threshold: L_max = (1/μ) * log(1/μ)
"""

import numpy as np
from typing import List, Tuple


def channel_capacity_simple(mutation_rate: float, genome_length: int) -> float:
    """
    Calculate maximum complexity using simple formula.
    
    Formula: C_max = -log2(μ * L)
    
    Args:
        mutation_rate: Per-base error rate (μ)
        genome_length: Number of bases (L)
    
    Returns:
        Maximum sustainable complexity (bits)
    
    TODO: Implement this!
    """
    pass


def channel_capacity_gaussian(mutation_rate: float, genome_length: int,
                               noise_variance: float = 0.1) -> float:
    """
    Calculate capacity for Gaussian noise channel.
    
    Formula: C = 0.5 * log2(1 / (2πe * σ²))
    
    TODO: Implement this!
    """
    pass


def eigen_error_threshold(mutation_rate: float) -> float:
    """
    Calculate Eigen's error threshold - maximum genome length.
    
    Formula: L_max = (1/μ) * ln(1/μ)
    
    Beyond this length, information is lost faster than gained!
    
    TODO: Implement this!
    """
    pass


def fidelity_complexity_curve(mutation_rates: np.ndarray, 
                               genome_length: int) -> np.ndarray:
    """
    Generate trade-off curve: C_max vs μ.
    
    Args:
        mutation_rates: Array of mutation rates to test
        genome_length: Fixed genome size
    
    Returns:
        Array of C_max values
    
    TODO: Implement this!
    Hint: Use channel_capacity_simple for each μ
    """
    pass


def required_fidelity(target_complexity: float, genome_length: int) -> float:
    """
    Find mutation rate needed for target complexity.
    
    Inverse problem: Given C_max, find μ.
    
    From C_max = -log2(μ * L), solve for μ:
    μ = 2^(-C_max) / L
    
    TODO: Implement this!
    """
    pass


def compare_organisms_capacity() -> dict:
    """
    Compare channel capacity for different organisms.
    
    Returns dictionary with organism data.
    
    TODO: Implement this!
    Use realistic biological parameters.
    """
    organisms = {
        'RNA Virus': {'mu': 1e-4, 'L': 1e4},
        'Bacteria': {'mu': 1e-6, 'L': 1e6},
        'Human': {'mu': 1e-9, 'L': 1e9},
    }
    
    results = {}
    for name, params in organisms.items():
        # TODO: Calculate C_max, L_max, etc.
        pass
    
    return results


# ============================================================================
# TEST CASES
# ============================================================================

def run_tests():
    """Run all test cases."""
    print("="*70)
    print("TESTING EXERCISE 3: Channel Capacity")
    print("="*70)
    
    # Test 1: Simple capacity
    print("\n[Test 1] Simple channel capacity...")
    C_max = channel_capacity_simple(mu=1e-6, genome_length=1e6)
    expected = -np.log2(1e-6 * 1e6)
    assert abs(C_max - expected) < 0.01, f"Expected {expected}, got {C_max}"
    print(f"✅ C_max = {C_max:.4f} bits")
    
    # Test 2: Error threshold
    print("\n[Test 2] Eigen's error threshold...")
    L_max = eigen_error_threshold(mu=1e-6)
    expected_L = (1/1e-6) * np.log(1/1e-6)
    assert abs(L_max - expected_L) < 1.0, f"Expected {expected_L}, got {L_max}"
    print(f"✅ L_max = {L_max:.0f} bases")
    
    # Test 3: Trade-off curve
    print("\n[Test 3] Fidelity-complexity curve...")
    mu_values = np.logspace(-9, -3, 10)
    C_values = fidelity_complexity_curve(mu_values, genome_length=1e6)
    
    assert len(C_values) == len(mu_values)
    assert C_values[0] > C_values[-1], "Lower μ should give higher C_max"
    print(f"✅ Curve generated: {len(C_values)} points")
    
    # Test 4: Required fidelity
    print("\n[Test 4] Required fidelity...")
    target_C = 20.0
    mu_required = required_fidelity(target_C, genome_length=1e6)
    
    # Verify: does this μ give target C?
    C_check = channel_capacity_simple(mu_required, 1e6)
    assert abs(C_check - target_C) < 0.1, f"Expected {target_C}, got {C_check}"
    print(f"✅ Required μ = {mu_required:.2e} for C_max = {target_C}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


def demo():
    """Demonstrate channel capacity."""
    print("\n" + "="*70)
    print("DEMO: Channel Capacity & Error Threshold")
    print("="*70)
    
    # Compare organisms
    print("\nOrganism Comparison:")
    print("-" * 70)
    print(f"{'Organism':<15} {'μ':<15} {'L':<15} {'C_max':<15} {'L_max':<15}")
    print("-" * 70)
    
    organisms = {
        'Virus': {'mu': 1e-4, 'L': 1e4},
        'Bacteria': {'mu': 1e-6, 'L': 1e6},
        'Insect': {'mu': 1e-8, 'L': 1e8},
        'Human': {'mu': 1e-9, 'L': 1e9},
    }
    
    for name, params in organisms.items():
        C_max = channel_capacity_simple(params['mu'], params['L'])
        L_max = eigen_error_threshold(params['mu'])
        
        print(f"{name:<15} {params['mu']:<15.0e} {params['L']:<15.0e} "
              f"{C_max:<15.2f} {L_max:<15.0e}")
    
    print("-" * 70)
    print("\n💡 Key insight: Better copying (lower μ) → Higher capacity!")
    print("💡 But there's always a limit (error threshold)!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        try:
            run_tests()
            demo()
        except (AssertionError, TypeError) as e:
            print(f"\n❌ Test failed: {e}")
            sys.exit(1)
        except NotImplementedError:
            print("\n⚠️  Functions not yet implemented!")
            sys.exit(1)
