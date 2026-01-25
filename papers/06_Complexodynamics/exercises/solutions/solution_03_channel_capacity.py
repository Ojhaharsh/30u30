"""
Solution to Exercise 3: Channel Capacity
"""

import numpy as np
from typing import List, Tuple


def channel_capacity_simple(mutation_rate: float, genome_length: int) -> float:
    """Calculate maximum complexity using simple formula."""
    if mutation_rate * genome_length >= 1:
        return 0.0  # Error catastrophe
    return -np.log2(mutation_rate * genome_length)


def channel_capacity_gaussian(mutation_rate: float, genome_length: int,
                               noise_variance: float = 0.1) -> float:
    """Calculate capacity for Gaussian noise channel."""
    sigma_squared = noise_variance
    return 0.5 * np.log2(1 / (2 * np.pi * np.e * sigma_squared))


def eigen_error_threshold(mutation_rate: float) -> float:
    """Calculate Eigen's error threshold."""
    if mutation_rate <= 0:
        return np.inf
    return (1 / mutation_rate) * np.log(1 / mutation_rate)


def fidelity_complexity_curve(mutation_rates: np.ndarray, 
                               genome_length: int) -> np.ndarray:
    """Generate trade-off curve."""
    return np.array([channel_capacity_simple(mu, genome_length) 
                     for mu in mutation_rates])


def required_fidelity(target_complexity: float, genome_length: int) -> float:
    """Find mutation rate for target complexity."""
    # From C_max = -log2(μ * L)
    # μ = 2^(-C_max) / L
    return 2**(-target_complexity) / genome_length


def compare_organisms_capacity() -> dict:
    """Compare channel capacity for different organisms."""
    organisms = {
        'RNA Virus': {'mu': 1e-4, 'L': 1e4},
        'Bacteria': {'mu': 1e-6, 'L': 1e6},
        'Human': {'mu': 1e-9, 'L': 1e9},
    }
    
    results = {}
    for name, params in organisms.items():
        C_max = channel_capacity_simple(params['mu'], params['L'])
        L_max = eigen_error_threshold(params['mu'])
        at_threshold = params['L'] / L_max
        
        results[name] = {
            'C_max': C_max,
            'L_max': L_max,
            'at_threshold': at_threshold,
            'safe': at_threshold < 0.5
        }
    
    return results


def run_tests():
    """Run all test cases."""
    print("="*70)
    print("TESTING SOLUTION 3: Channel Capacity")
    print("="*70)
    
    print("\n[Test 1] Simple capacity...")
    C_max = channel_capacity_simple(1e-6, 1e6)
    expected = -np.log2(1e-6 * 1e6)
    assert abs(C_max - expected) < 0.01
    print(f"✅ C_max = {C_max:.4f}")
    
    print("\n[Test 2] Error threshold...")
    L_max = eigen_error_threshold(1e-6)
    expected_L = (1/1e-6) * np.log(1/1e-6)
    assert abs(L_max - expected_L) < 1.0
    print(f"✅ L_max = {L_max:.0f}")
    
    print("\n[Test 3] Trade-off curve...")
    mu_values = np.logspace(-9, -3, 10)
    C_values = fidelity_complexity_curve(mu_values, 1e6)
    assert len(C_values) == 10
    assert C_values[0] > C_values[-1]
    print("✅ Curve generated")
    
    print("\n[Test 4] Required fidelity...")
    mu_req = required_fidelity(20.0, 1e6)
    C_check = channel_capacity_simple(mu_req, 1e6)
    assert abs(C_check - 20.0) < 0.1
    print(f"✅ μ = {mu_req:.2e}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


def demo():
    """Demonstrate solution."""
    print("\n" + "="*70)
    print("DEMO: Channel Capacity Solution")
    print("="*70)
    
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


if __name__ == '__main__':
    run_tests()
    demo()
