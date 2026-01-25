"""
Solution to Exercise 2: Information Flow
"""

import numpy as np
from typing import Tuple


class InformationFlowCalculator:
    """Calculate information gain and loss in evolving populations."""
    
    def __init__(self, mutation_rate: float, genome_length: int, 
                 selection_strength: float, alphabet_size: int = 4):
        self.mu = mutation_rate
        self.L = genome_length
        self.beta = selection_strength
        self.alphabet_size = alphabet_size
    
    def information_gain(self, fitness_variance: float) -> float:
        """
        Calculate I_E (information gain from selection).
        
        Approximation: I_E ≈ β * Var(fitness)
        """
        return self.beta * fitness_variance
    
    def information_loss(self) -> float:
        """
        Calculate I_L (information loss from mutation).
        
        Formula: I_L = μ * L * log2(alphabet_size)
        """
        return self.mu * self.L * np.log2(self.alphabet_size)
    
    def net_flow(self, fitness_variance: float) -> float:
        """Calculate net information flow (dC/dt)."""
        return self.information_gain(fitness_variance) - self.information_loss()
    
    def equilibrium_complexity(self) -> float:
        """
        Calculate equilibrium complexity where I_E = I_L.
        
        Using simple approximation: C_eq = -log2(μ * L) / L
        """
        if self.mu * self.L >= 1:
            return 0.0  # Error catastrophe
        return -np.log2(self.mu * self.L) / self.L * 1e6  # Bits per site
    
    def is_at_equilibrium(self, fitness_variance: float, threshold: float = 0.01) -> bool:
        """Check if system is at equilibrium."""
        net = self.net_flow(fitness_variance)
        return abs(net) < threshold


def simulate_information_flow(generations: int, mutation_rate: float,
                               genome_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate information flow over time."""
    calc = InformationFlowCalculator(mutation_rate, genome_length, 0.01)
    
    time_points = np.arange(generations)
    I_E_history = []
    I_L_history = []
    
    # Assume fitness variance decreases exponentially as complexity increases
    initial_variance = 100.0
    decay_rate = 0.01
    
    for gen in range(generations):
        fitness_variance = initial_variance * np.exp(-decay_rate * gen)
        I_E = calc.information_gain(fitness_variance)
        I_L = calc.information_loss()
        
        I_E_history.append(I_E)
        I_L_history.append(I_L)
    
    return time_points, np.array(I_E_history), np.array(I_L_history)


def find_equilibrium_parameters(target_complexity: float, genome_length: int,
                                 alphabet_size: int = 4) -> Tuple[float, float]:
    """Find mutation rate for target equilibrium complexity."""
    # From C_max = -log2(μ * L) / L
    # Solve for μ: μ = 2^(-C_max * L) / L
    
    C_max_total = target_complexity * genome_length / 1e6  # Convert bits/site to total bits
    mu = 2**(-C_max_total) / genome_length
    
    # Selection strength affects τ, not C_max
    # Use moderate value
    beta = 0.01
    
    return mu, beta


def run_tests():
    """Run all test cases."""
    print("="*70)
    print("TESTING SOLUTION 2: Information Flow")
    print("="*70)
    
    print("\n[Test 1] Basic calculation...")
    calc = InformationFlowCalculator(1e-6, 1000000, 0.01, 4)
    
    I_L = calc.information_loss()
    assert I_L > 0
    assert abs(I_L - 2.0) < 0.01
    print(f"✅ I_L = {I_L:.6f}")
    
    I_E = calc.information_gain(10.0)
    assert I_E > 0
    print(f"✅ I_E = {I_E:.6f}")
    
    print("\n[Test 2] Net flow...")
    net = calc.net_flow(10.0)
    assert abs(net - (I_E - I_L)) < 1e-10
    print("✅ Net flow correct")
    
    print("\n[Test 3] Equilibrium...")
    eq_variance = I_L / calc.beta
    assert calc.is_at_equilibrium(eq_variance, threshold=0.1)
    print("✅ Equilibrium detection works")
    
    print("\n[Test 4] Mutation effects...")
    low_mu = InformationFlowCalculator(1e-9, 1000000, 0.01)
    high_mu = InformationFlowCalculator(1e-4, 1000000, 0.01)
    assert low_mu.information_loss() < high_mu.information_loss()
    print("✅ Mutation effects correct")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


def demo():
    """Demonstrate solution."""
    print("\n" + "="*70)
    print("DEMO: Information Flow Solution")
    print("="*70)
    
    calc = InformationFlowCalculator(1e-6, 1000000, 0.01)
    
    print("\nInformation Flow Analysis:")
    print("-" * 70)
    print(f"{'Variance':<15} {'I_E':<15} {'I_L':<15} {'Net':<15} {'Status':<15}")
    print("-" * 70)
    
    I_L = calc.information_loss()
    for var in [1.0, 10.0, 50.0, 100.0, 200.0]:
        I_E = calc.information_gain(var)
        net = I_E - I_L
        status = "Eq" if abs(net) < 0.1 else ("↑" if net > 0 else "↓")
        print(f"{var:<15.1f} {I_E:<15.6f} {I_L:<15.6f} {net:<15.6f} {status:<15}")
    print("-" * 70)


if __name__ == '__main__':
    run_tests()
    demo()
