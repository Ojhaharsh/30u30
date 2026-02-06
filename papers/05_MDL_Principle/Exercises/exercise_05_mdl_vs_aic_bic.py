"""
Exercise 5: MDL vs AIC vs BIC - The Ultimate Showdown
======================================================

Three model selection criteria, one goal: Find the true model!

MDL (Minimum Description Length):
    Score = L(Model) + L(Data|Model)
    → Information-theoretic penalty
    → No arbitrary constants

AIC (Akaike Information Criterion):
    Score = 2k - 2ln(L)
    → Penalty = 2 per parameter
    → Derived from prediction risk

BIC (Bayesian Information Criterion):
    Score = k·ln(n) - 2ln(L)
    → Penalty = ln(n) per parameter
    → Derived from Bayesian model selection

The Key Question:
-----------------
When do these methods agree? When do they disagree?
And which one should you trust?

Spoiler: It depends on your goals!
- AIC: Best for prediction
- BIC: Best for finding "true" model
- MDL: Best for compression/understanding
"""

import numpy as np
from typing import Dict, Tuple, List


def compute_aic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """
    Compute AIC (Akaike Information Criterion).
    
    AIC = 2k - 2ln(L)
    
    where k = number of parameters, L = maximum likelihood
    
    Lower is better!
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree
        
    Returns:
        AIC score (lower = better)
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 2*x + 1 + np.random.randn(50) * 0.5
        >>> aic_1 = compute_aic(x, y, 1)  # Should be low (good fit)
        >>> aic_5 = compute_aic(x, y, 5)  # Should be higher (overfit)
    """
    # TODO: Implement AIC
    # Steps:
    #   1. k = degree + 2 (polynomial coefficients + variance)
    #   2. Fit polynomial
    #   3. Compute ML estimate of sigma: sqrt(RSS/n)
    #   4. Compute log-likelihood: -n/2 * (log(2π σ²) + 1)
    #   5. Return 2k - 2*log_likelihood
    
    pass  # Your code here


def compute_bic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """
    Compute BIC (Bayesian Information Criterion).
    
    BIC = k·ln(n) - 2ln(L)
    
    BIC has a stronger penalty than AIC for large samples.
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree
        
    Returns:
        BIC score (lower = better)
    """
    # TODO: Implement BIC
    # Same as AIC, but penalty is k * log(n) instead of 2k
    
    pass  # Your code here


def compute_mdl(x: np.ndarray, y: np.ndarray, degree: int,
                precision: int = 32) -> float:
    """
    Compute two-part MDL score.
    
    MDL = L(Model) + L(Data|Model)
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree
        precision: Bits per coefficient
        
    Returns:
        MDL score in bits (lower = better)
    """
    # TODO: Implement MDL (copy from exercise 1 or implement fresh)
    
    pass  # Your code here


def select_with_all_criteria(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Dict]:
    """
    Run all three model selection criteria and compare.
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to consider
        
    Returns:
        Dictionary with results for each method:
        {
            'mdl': {'scores': {...}, 'best': int},
            'aic': {'scores': {...}, 'best': int},
            'bic': {'scores': {...}, 'best': int}
        }
    """
    # TODO: Compute all scores and find best for each
    
    pass  # Your code here


def monte_carlo_comparison(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """
    Compare accuracy of MDL, AIC, and BIC over many trials.
    
    This is the definitive test: Which method most often
    finds the TRUE underlying model?
    
    Args:
        true_degree: The actual polynomial degree
        n_samples: Data points per trial
        noise_std: Noise level
        n_trials: Number of experiments
        
    Returns:
        Dictionary with accuracy for each method:
        {'mdl_accuracy': 0.85, 'aic_accuracy': 0.72, 'bic_accuracy': 0.88}
    """
    # TODO: Implement Monte Carlo comparison
    # Steps:
    #   1. For each trial:
    #      a. Generate random polynomial of true_degree
    #      b. Add noise
    #      c. Apply each criterion
    #      d. Check which found the true degree
    #   2. Compute accuracy for each
    
    pass  # Your code here


def analyze_sample_size_effect(
    true_degree: int = 2,
    sample_sizes: List[int] = [20, 50, 100, 200, 500],
    noise_std: float = 1.5,
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """
    Analyze how sample size affects each criterion's accuracy.
    
    Key insight:
    - AIC tends to overfit with large samples
    - BIC gets more accurate with large samples
    - MDL should be relatively stable
    
    Returns:
        Dictionary mapping method -> list of accuracies for each sample size
    """
    # TODO: Implement sample size analysis
    
    pass  # Your code here


def analyze_noise_level_effect(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_levels: List[float] = [0.5, 1.0, 2.0, 4.0],
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """
    Analyze how noise level affects each criterion's accuracy.
    
    Key insight: All methods struggle with high noise,
    but some are more robust than others.
    
    Returns:
        Dictionary mapping method -> list of accuracies for each noise level
    """
    # TODO: Implement noise level analysis
    
    pass  # Your code here


def when_do_they_disagree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, any]:
    """
    Find cases where the methods disagree and analyze why.
    
    Returns analysis of the disagreement, including:
    - Which methods picked which degree
    - The score differences
    - Possible explanation
    """
    # TODO: Implement disagreement analysis
    
    pass  # Your code here


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_polynomial_data(
    degree: int,
    n_samples: int,
    noise_std: float,
    x_range: Tuple[float, float] = (0, 10),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic polynomial data.
    
    Returns:
        Tuple of (x, y_noisy, true_coefficients)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(x_range[0], x_range[1], n_samples)
    
    # Random coefficients
    true_coeffs = np.random.randn(degree + 1) * 2
    poly = np.poly1d(true_coeffs)
    
    y_true = poly(x)
    y_noisy = y_true + np.random.randn(n_samples) * noise_std
    
    return x, y_noisy, true_coeffs


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_implementation():
    """Test the criterion comparison implementation."""
    
    print("Testing MDL vs AIC vs BIC Comparison")
    print("=" * 55)
    
    np.random.seed(42)
    
    # Generate test data
    true_degree = 2
    x, y, true_coeffs = generate_polynomial_data(
        degree=true_degree,
        n_samples=50,
        noise_std=1.5,
        seed=42
    )
    
    print(f"\nTest data: n=50, true degree={true_degree}")
    
    # Test 1: Individual criteria
    print("\n1. Testing individual criteria...")
    try:
        aic_2 = compute_aic(x, y, 2)
        bic_2 = compute_bic(x, y, 2)
        mdl_2 = compute_mdl(x, y, 2)
        
        print(f"   AIC(degree=2) = {aic_2:.2f}")
        print(f"   BIC(degree=2) = {bic_2:.2f}")
        print(f"   MDL(degree=2) = {mdl_2:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 2: Full selection
    print("\n2. Testing select_with_all_criteria()...")
    try:
        results = select_with_all_criteria(x, y, max_degree=8)
        
        print(f"   MDL selects: degree {results['mdl']['best']}")
        print(f"   AIC selects: degree {results['aic']['best']}")
        print(f"   BIC selects: degree {results['bic']['best']}")
        print(f"   True degree: {true_degree}")
        
        # Count how many got it right
        correct = sum([
            results['mdl']['best'] == true_degree,
            results['aic']['best'] == true_degree,
            results['bic']['best'] == true_degree
        ])
        print(f"   {correct}/3 methods correct")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 3: Monte Carlo
    print("\n3. Testing monte_carlo_comparison()...")
    try:
        mc_results = monte_carlo_comparison(
            true_degree=2,
            n_samples=50,
            noise_std=1.5,
            n_trials=50
        )
        
        print(f"   MDL accuracy: {mc_results['mdl_accuracy']*100:.1f}%")
        print(f"   AIC accuracy: {mc_results['aic_accuracy']*100:.1f}%")
        print(f"   BIC accuracy: {mc_results['bic_accuracy']*100:.1f}%")
        
        # Find winner
        winner = max(['mdl', 'aic', 'bic'],
                     key=lambda m: mc_results[f'{m}_accuracy'])
        print(f"   Winner: {winner.upper()}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 4: Sample size effect
    print("\n4. Testing analyze_sample_size_effect()...")
    try:
        size_results = analyze_sample_size_effect(
            true_degree=2,
            sample_sizes=[20, 50, 100],
            n_trials=30
        )
        
        for method in ['mdl', 'aic', 'bic']:
            accs = size_results[method]
            print(f"   {method.upper()}: n=20→{accs[0]*100:.0f}%, n=50→{accs[1]*100:.0f}%, n=100→{accs[2]*100:.0f}%")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [NOTE] Skipped: {e}")
    
    # Test 5: Disagreement analysis
    print("\n5. Testing when_do_they_disagree()...")
    try:
        disagreement = when_do_they_disagree(x, y, max_degree=8)
        
        if disagreement.get('methods_agree', False):
            print("   All methods agree on this dataset")
        else:
            print(f"   Methods disagree!")
            print(f"   Analysis: {disagreement.get('explanation', 'N/A')}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [NOTE] Skipped: {e}")
    
    print("\n" + "=" * 55)
    print("Testing complete!")
    print("\nKey takeaways:")
    print("- AIC: Optimizes prediction, may overfit with large n")
    print("- BIC: Consistent, gets better with more data")
    print("- MDL: Information-theoretic, no arbitrary constants")


if __name__ == "__main__":
    test_implementation()
