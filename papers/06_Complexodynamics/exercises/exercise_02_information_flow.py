"""
Exercise 2: Information Flow (⭐⭐ Medium - 45 minutes)

Calculate information gain (I_E) and loss (I_L) rates.

Key formulas:
    I_E = KL(p_after || p_before)  # Information from selection
    I_L = μ * L * log2(alphabet_size)  # Information from mutation

Equilibrium: I_E = I_L → dC/dt = 0
"""

import numpy as np
from typing import Tuple


class InformationFlowCalculator:
    """Calculate information gain and loss in evolving populations."""
    
    def __init__(self, mutation_rate: float, genome_length: int, 
                 selection_strength: float, alphabet_size: int = 4):
        """
        Initialize calculator.
        
        Args:
            mutation_rate: Per-base mutation probability (μ)
            genome_length: Number of bases (L)
            selection_strength: Fitness advantage (β)
            alphabet_size: Number of symbols (4 for DNA)
        """
        self.mu = mutation_rate
        self.L = genome_length
        self.beta = selection_strength
        self.alphabet_size = alphabet_size
    
    def information_gain(self, fitness_variance: float) -> float:
        """
        Calculate I_E (information gain from selection).
        
        Args:
            fitness_variance: Var(fitness) in population
            
        Returns:
            Information gain in bits/generation
        
        Formula:
            I_E ≈ β * Var(fitness)
        
        TODO: Implement this!
        """
        pass
    
    def information_loss(self) -> float:
        """
        Calculate I_L (information loss from mutation).
        
        Returns:
            Information loss in bits/generation
        
        Formula:
            I_L = μ * L * log2(alphabet_size)
        
        For DNA: I_L = μ * L * 2
        
        TODO: Implement this!
        """
        pass
    
    def net_flow(self, fitness_variance: float) -> float:
        """
        Calculate net information flow (dC/dt).
        
        Returns:
            dC/dt = I_E - I_L
        
        TODO: Implement this!
        """
        pass
    
    def equilibrium_complexity(self) -> float:
        """
        Calculate equilibrium complexity C_eq where I_E = I_L.
        
        Returns:
            C_eq in bits
        
        TODO: Implement this!
        Hint: At equilibrium, information gain equals loss.
        """
        pass
    
    def is_at_equilibrium(self, fitness_variance: float, threshold: float = 0.01) -> bool:
        """
        Check if system is at equilibrium.
        
        Args:
            fitness_variance: Current population fitness variance
            threshold: Tolerance for |I_E - I_L|
        
        Returns:
            True if |I_E - I_L| < threshold
        
        TODO: Implement this!
        """
        pass


def simulate_information_flow(generations: int, mutation_rate: float,
                               genome_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate information flow over time.
    
    Args:
        generations: Number of generations to simulate
        mutation_rate: Per-base mutation rate
        genome_length: Genome size
    
    Returns:
        Tuple of (time_points, I_E_history, I_L_history)
    
    TODO: Implement this!
    
    Hints:
    1. Create InformationFlowCalculator
    2. Loop over generations
    3. Calculate I_E and I_L at each time point
    4. Assume fitness_variance decreases as complexity increases
    """
    pass


def find_equilibrium_parameters(target_complexity: float, genome_length: int,
                                 alphabet_size: int = 4) -> Tuple[float, float]:
    """
    Find mutation rate and selection strength for target equilibrium complexity.
    
    Args:
        target_complexity: Desired C_eq (bits)
        genome_length: Genome size
        alphabet_size: Number of symbols
    
    Returns:
        Tuple of (mutation_rate, selection_strength)
    
    TODO: Implement this!
    
    Hints:
    1. Use C_max formula: C_max = -log2(μ * L) / L * scale
    2. Solve for μ given target C
    3. Selection strength affects τ, not C_max
    """
    pass


# ============================================================================
# TEST CASES
# ============================================================================

def run_tests():
    """Run all test cases."""
    print("="*70)
    print("TESTING EXERCISE 2: Information Flow")
    print("="*70)
    
    # Test 1: Basic calculation
    print("\n[Test 1] Basic I_E and I_L calculation...")
    calc = InformationFlowCalculator(
        mutation_rate=1e-6,
        genome_length=1000000,
        selection_strength=0.01,
        alphabet_size=4
    )
    
    I_L = calc.information_loss()
    assert I_L > 0, "I_L should be positive"
    assert I_L == 1e-6 * 1000000 * 2, f"Expected 2.0, got {I_L}"
    print(f"✅ I_L = {I_L:.6f} bits/generation")
    
    I_E = calc.information_gain(fitness_variance=10.0)
    assert I_E > 0, "I_E should be positive"
    print(f"✅ I_E = {I_E:.6f} bits/generation")
    
    # Test 2: Net flow
    print("\n[Test 2] Net information flow...")
    net = calc.net_flow(fitness_variance=10.0)
    assert net == I_E - I_L, "Net flow should be I_E - I_L"
    print(f"✅ Net flow = {net:.6f} bits/generation")
    
    # Test 3: Equilibrium detection
    print("\n[Test 3] Equilibrium detection...")
    # High variance → I_E > I_L → not at equilibrium
    assert not calc.is_at_equilibrium(fitness_variance=100.0), \
        "Should not be at equilibrium with high variance"
    
    # Find variance where I_E ≈ I_L
    # I_E = β * Var → Var = I_L / β
    eq_variance = I_L / calc.beta
    assert calc.is_at_equilibrium(fitness_variance=eq_variance, threshold=0.1), \
        "Should be at equilibrium when I_E = I_L"
    print("✅ Equilibrium detection works")
    
    # Test 4: Different mutation rates
    print("\n[Test 4] Mutation rate effects...")
    low_mu = InformationFlowCalculator(1e-9, 1000000, 0.01)
    high_mu = InformationFlowCalculator(1e-4, 1000000, 0.01)
    
    assert low_mu.information_loss() < high_mu.information_loss(), \
        "Lower mutation rate should have less information loss"
    print("✅ Mutation rate effects correct")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


def demo():
    """Demonstrate information flow."""
    print("\n" + "="*70)
    print("DEMO: Information Flow Dynamics")
    print("="*70)
    
    # Create calculator
    calc = InformationFlowCalculator(
        mutation_rate=1e-6,
        genome_length=1000000,
        selection_strength=0.01
    )
    
    # Show flow for different fitness variances
    print("\nInformation Flow vs Fitness Variance:")
    print("-" * 70)
    print(f"{'Variance':<15} {'I_E':<15} {'I_L':<15} {'Net (dC/dt)':<15} {'Status':<15}")
    print("-" * 70)
    
    I_L = calc.information_loss()
    variances = [1.0, 10.0, 50.0, 100.0, 200.0]
    
    for var in variances:
        I_E = calc.information_gain(var)
        net = I_E - I_L
        status = "Equilibrium" if abs(net) < 0.1 else ("Growing" if net > 0 else "Shrinking")
        print(f"{var:<15.1f} {I_E:<15.6f} {I_L:<15.6f} {net:<15.6f} {status:<15}")
    
    print("-" * 70)
    print("\n💡 Equilibrium occurs when I_E = I_L (net flow ≈ 0)")


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
