"""
Exercise 4: NML (Normalized Maximum Likelihood) Complexity
==========================================================

NML is the "gold standard" of MDL, but harder to compute.

The key idea: Penalize models based on how many different 
datasets they could POTENTIALLY explain well.

The Restaurant Analogy:
-----------------------
- Simple model (degree 1): Like a restaurant with a tiny menu
  → Can only make a few things, but does them well
  
- Complex model (degree 9): Like a restaurant with 500 dishes
  → Can make almost anything, but that's suspicious!
  
When a huge-menu restaurant happens to have your exact dish,
it's less impressive than a tiny-menu restaurant having it.

NML Complexity = log(# of dishes the restaurant can make well)

Mathematical form:
    COMP(M) = log Σ_z max_θ P(z | θ, M)
    
Where we sum over ALL possible datasets z.
"""

import numpy as np
from typing import Tuple, Dict


def estimate_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = (-100, 100),
    n_samples: int = 1000
) -> float:
    """
    Estimate NML complexity for polynomial models using Monte Carlo.
    
    The NML complexity measures how many different datasets the model
    can explain well. Higher complexity = more flexible = needs more penalty.
    
    Algorithm:
    1. Sample many random datasets (y values)
    2. For each, fit the polynomial (get max likelihood)
    3. Average the max-likelihood values
    4. Take log → complexity
    
    Args:
        x: Input values (fixed design points)
        degree: Polynomial degree
        y_range: Range of possible y values for sampling
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated NML complexity in bits
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> comp_1 = estimate_nml_complexity_polynomial(x, 1)  # Low
        >>> comp_5 = estimate_nml_complexity_polynomial(x, 5)  # Higher
        >>> assert comp_5 > comp_1, "Higher degree = higher complexity"
    """
    # TODO: Implement NML complexity estimation
    # Steps:
    #   1. For each of n_samples iterations:
    #      a. Sample random y values from y_range
    #      b. Fit polynomial to (x, y_random)
    #      c. Compute maximum log-likelihood
    #   2. Use logsumexp trick to compute log of average
    #   3. Convert to bits (divide by log(2))
    
    pass  # Your code here


def compute_nml_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """
    Compute NML-based MDL score.
    
    NML MDL = -log P(y | θ̂, M) + COMP(M)
            = Fit term + Complexity term
    
    Unlike two-part code, NML doesn't require specifying precision.
    The complexity penalty emerges naturally from information theory.
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree
        y_range: Range for complexity estimation (auto if None)
        
    Returns:
        Tuple of (total_nml, fit_term, complexity_term)
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> total, fit, comp = compute_nml_mdl(x, y, 2)
        >>> print(f"NML = {fit:.1f} (fit) + {comp:.1f} (complexity) = {total:.1f}")
    """
    # TODO: Implement NML MDL
    # Steps:
    #   1. Fit polynomial and compute negative log-likelihood
    #   2. Estimate NML complexity
    #   3. Return sum and components
    
    pass  # Your code here


def compare_nml_vs_two_part(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, int]:
    """
    Compare NML and Two-Part Code model selection.
    
    They often agree, but NML is more principled (no precision parameter).
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to try
        
    Returns:
        Dictionary with 'nml_best' and 'two_part_best' degrees
    """
    # TODO: Implement comparison
    
    pass  # Your code here


def visualize_complexity_vs_degree(x: np.ndarray, max_degree: int = 10):
    """
    Show how NML complexity grows with model flexibility.
    
    This visualization demonstrates the key insight:
    More flexible models have higher complexity → more penalty.
    """
    # TODO: Compute and print complexity for each degree
    # (Visualization code is optional if matplotlib not available)
    
    pass  # Your code here


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def logsumexp(log_values: np.ndarray) -> float:
    """
    Compute log(sum(exp(values))) in a numerically stable way.
    
    This is essential for NML computation to avoid overflow.
    
    Args:
        log_values: Array of log values
        
    Returns:
        log(sum(exp(log_values)))
    """
    # TODO: Implement numerically stable logsumexp
    # Hint: log(Σ exp(xi)) = max(xi) + log(Σ exp(xi - max))
    
    pass  # Your code here


def gaussian_log_likelihood(
    y: np.ndarray,
    predictions: np.ndarray,
    sigma: float = None
) -> float:
    """
    Compute Gaussian log-likelihood.
    
    Args:
        y: Actual values
        predictions: Model predictions
        sigma: Standard deviation (estimated from residuals if None)
        
    Returns:
        Log-likelihood (not in bits, in nats)
    """
    # TODO: Implement Gaussian log-likelihood
    
    pass  # Your code here


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_implementation():
    """Test NML complexity implementation."""
    
    print("Testing NML (Normalized Maximum Likelihood) Implementation")
    print("=" * 55)
    
    np.random.seed(42)
    
    x = np.linspace(0, 10, 50)
    
    # Test 1: NML complexity increases with degree
    print("\n1. Testing NML complexity vs degree...")
    try:
        complexities = {}
        for deg in [1, 2, 3, 5]:
            comp = estimate_nml_complexity_polynomial(x, deg, n_samples=200)
            complexities[deg] = comp
            print(f"   Degree {deg}: COMP = {comp:.2f} bits")
        
        # Complexity should generally increase with degree
        assert complexities[5] > complexities[1], "Higher degree should have higher complexity"
        print("   [ok] PASSED - Complexity increases with flexibility")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 2: logsumexp
    print("\n2. Testing logsumexp()...")
    try:
        values = np.array([1000, 1001, 1002])  # Would overflow with naive exp
        result = logsumexp(values)
        
        # Should be approximately 1002 + log(1 + e^-1 + e^-2) ≈ 1002.41
        assert 1002 < result < 1003, f"Expected ~1002.4, got {result}"
        print(f"   logsumexp([1000, 1001, 1002]) = {result:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 3: NML MDL score
    print("\n3. Testing compute_nml_mdl()...")
    try:
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # True degree 2
        
        nml_1 = compute_nml_mdl(x, y, 1)
        nml_2 = compute_nml_mdl(x, y, 2)
        nml_5 = compute_nml_mdl(x, y, 5)
        
        print(f"   NML(degree=1): {nml_1[0]:.1f} = {nml_1[1]:.1f} + {nml_1[2]:.1f}")
        print(f"   NML(degree=2): {nml_2[0]:.1f} = {nml_2[1]:.1f} + {nml_2[2]:.1f}")
        print(f"   NML(degree=5): {nml_5[0]:.1f} = {nml_5[1]:.1f} + {nml_5[2]:.1f}")
        
        if nml_2[0] < nml_1[0] and nml_2[0] < nml_5[0]:
            print("   [ok] PASSED - NML correctly prefers degree 2!")
        else:
            print("   [NOTE] Score ordering varies (NML is stochastic)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 4: Compare NML vs Two-Part
    print("\n4. Testing compare_nml_vs_two_part()...")
    try:
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
        
        result = compare_nml_vs_two_part(x, y, max_degree=6)
        
        print(f"   NML selects: degree {result['nml_best']}")
        print(f"   Two-Part selects: degree {result['two_part_best']}")
        
        if result['nml_best'] == result['two_part_best']:
            print("   [ok] Methods agree!")
        else:
            print("   [NOTE] Methods disagree (both are valid)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 5: Complexity visualization
    print("\n5. Testing visualize_complexity_vs_degree()...")
    try:
        visualize_complexity_vs_degree(x, max_degree=5)
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [NOTE] Skipped: {e}")
    
    print("\n" + "=" * 55)
    print("Testing complete!")
    print("\nKey insight: NML complexity grows with model flexibility.")
    print("This is why complex models get penalized more!")


if __name__ == "__main__":
    test_implementation()
