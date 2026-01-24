"""
Exercise 1: Two-Part Code MDL
==============================

The Two-Part Code is the foundation of MDL:
    L(total) = L(H) + L(D|H)

Where:
    L(H)   = bits to describe the MODEL (hypothesis)
    L(D|H) = bits to describe the DATA given the model

Your task: Implement the two-part code for polynomial regression.

The Spy Analogy:
----------------
You're a spy sending temperature data to HQ. You can either:
1. Send every number individually (naive, expensive)
2. Send a formula + small corrections (smart, cheap!)

The two-part code formalizes this: Model + Residuals = Total message.
"""

import numpy as np
from typing import Tuple


def universal_code_length(n: int) -> float:
    """
    Compute the universal code length for a positive integer.
    
    This is the "fair" way to encode integers without knowing their range.
    Uses the log-star function: log(n) + log(log(n)) + ... (positive terms only)
    
    Args:
        n: Positive integer to encode
        
    Returns:
        Code length in bits
        
    Example:
        >>> universal_code_length(1)  # Should be small
        >>> universal_code_length(100)  # Should be larger
    """
    # TODO: Implement universal integer coding
    # Hint: Start with log2(c0) where c0 ≈ 2.865
    # Then add log2(n) + log2(log2(n)) + ... while positive
    
    pass  # Your code here


def model_description_length(degree: int, precision_bits: int = 32) -> float:
    """
    Compute L(H): bits needed to describe a polynomial model.
    
    We need to encode:
    1. The degree of the polynomial
    2. The coefficients (precision_bits each, plus sign)
    
    Args:
        degree: Polynomial degree (0 = constant, 1 = linear, etc.)
        precision_bits: Bits per coefficient
        
    Returns:
        Total bits for model description
        
    Example:
        >>> model_description_length(0)  # Constant: 1 coefficient
        >>> model_description_length(2)  # Quadratic: 3 coefficients
    """
    # TODO: Implement model cost computation
    # Hint: 
    #   - Degree cost: universal_code_length(degree + 1)
    #   - Coefficient cost: (degree + 1) * (precision_bits + 1)
    
    pass  # Your code here


def data_description_length(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> float:
    """
    Compute L(D|H): bits needed to describe data given the model.
    
    After fitting the polynomial, we encode the residuals (errors).
    Using a Gaussian model, each residual costs approximately:
        0.5 * log2(2π σ²) + error² / (2σ² ln(2))
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree to fit
        precision_bits: Bits for encoding sigma
        
    Returns:
        Total bits for data description
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 2*x + 1 + np.random.randn(50) * 0.1  # Almost linear
        >>> data_description_length(x, y, 1)  # Should be small (good fit)
        >>> data_description_length(x, y, 0)  # Should be larger (poor fit)
    """
    # TODO: Implement data cost computation
    # Steps:
    #   1. Fit polynomial: coeffs = np.polyfit(x, y, degree)
    #   2. Compute predictions: predictions = np.poly1d(coeffs)(x)
    #   3. Compute residuals: residuals = y - predictions
    #   4. Estimate sigma: sigma = np.std(residuals)
    #   5. Compute Gaussian NLL in bits
    #   6. Add cost to encode sigma
    
    pass  # Your code here


def two_part_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    Compute the complete Two-Part MDL score.
    
    This combines model description length and data description length.
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree
        precision_bits: Precision for encoding
        
    Returns:
        Tuple of (total_mdl, model_cost, data_cost)
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> total, model, data = two_part_mdl(x, y, 2)
        >>> print(f"Total: {total:.1f} = Model: {model:.1f} + Data: {data:.1f}")
    """
    # TODO: Combine model and data description lengths
    
    pass  # Your code here


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_implementation():
    """Test your implementation against expected behavior."""
    
    print("Testing Two-Part Code MDL Implementation")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Universal code length
    print("\n1. Testing universal_code_length()...")
    try:
        len_1 = universal_code_length(1)
        len_10 = universal_code_length(10)
        len_100 = universal_code_length(100)
        
        assert len_1 < len_10 < len_100, "Larger integers should have longer codes"
        assert len_1 > 0, "Code length should be positive"
        print(f"   universal_code_length(1) = {len_1:.2f}")
        print(f"   universal_code_length(10) = {len_10:.2f}")
        print(f"   universal_code_length(100) = {len_100:.2f}")
        print("   ✓ PASSED")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 2: Model description length
    print("\n2. Testing model_description_length()...")
    try:
        model_0 = model_description_length(0)
        model_2 = model_description_length(2)
        model_5 = model_description_length(5)
        
        assert model_0 < model_2 < model_5, "Higher degree = more parameters = longer code"
        print(f"   model_description_length(0) = {model_0:.2f}")
        print(f"   model_description_length(2) = {model_2:.2f}")
        print(f"   model_description_length(5) = {model_5:.2f}")
        print("   ✓ PASSED")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 3: Data description length
    print("\n3. Testing data_description_length()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x + np.random.randn(50) * 0.5  # Linear with low noise
        
        data_0 = data_description_length(x, y, 0)  # Constant (bad fit)
        data_1 = data_description_length(x, y, 1)  # Linear (good fit)
        
        assert data_1 < data_0, "Better fitting model should have lower data cost"
        print(f"   data_cost(degree=0) = {data_0:.2f} (poor fit)")
        print(f"   data_cost(degree=1) = {data_1:.2f} (good fit)")
        print("   ✓ PASSED")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 4: Two-part MDL
    print("\n4. Testing two_part_mdl()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # Quadratic
        
        mdl_1, m1, d1 = two_part_mdl(x, y, 1)
        mdl_2, m2, d2 = two_part_mdl(x, y, 2)
        mdl_5, m5, d5 = two_part_mdl(x, y, 5)
        
        # Degree 2 should have best (lowest) total MDL
        assert mdl_2 < mdl_1, "True degree (2) should beat underfit (1)"
        assert mdl_2 < mdl_5, "True degree (2) should beat overfit (5)"
        
        print(f"   MDL(degree=1) = {mdl_1:.1f} (model: {m1:.1f}, data: {d1:.1f})")
        print(f"   MDL(degree=2) = {mdl_2:.1f} (model: {m2:.1f}, data: {d2:.1f}) ← BEST")
        print(f"   MDL(degree=5) = {mdl_5:.1f} (model: {m5:.1f}, data: {d5:.1f})")
        print("   ✓ PASSED")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 5: Model selection
    print("\n5. Testing model selection...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # True degree = 2
        
        scores = {d: two_part_mdl(x, y, d)[0] for d in range(6)}
        best_degree = min(scores, key=scores.get)
        
        print(f"   MDL scores: {[f'{d}:{scores[d]:.0f}' for d in range(6)]}")
        print(f"   Selected degree: {best_degree}")
        print(f"   True degree: 2")
        
        if best_degree == 2:
            print("   ✓ PASSED - MDL found the true structure!")
        else:
            print(f"   ⚠ Selected {best_degree} instead of 2 (might be noise)")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    test_implementation()
