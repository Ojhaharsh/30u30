"""
Exercise 2: Prequential (Predictive) MDL
=========================================

Prequential MDL scores a model by how well it predicts UNSEEN data.

The key idea:
1. Start with a few data points
2. Fit model to those points
3. PREDICT the next point (before seeing it)
4. Record the "surprise" (how wrong was the prediction?)
5. Add the point to training data, repeat

Total surprise = Prequential MDL score

The Weather Forecaster Analogy:
-------------------------------
A weather forecaster makes predictions every day.
We score them by how surprised they are by actual outcomes.
A good forecaster is rarely very surprised → Low prequential score.

Advantage over Two-Part Code:
- No need to specify parameter precision
- More robust for time series
- Naturally penalizes complexity
"""

import numpy as np
from typing import Tuple, List


def predict_next_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: float,
    degree: int
) -> Tuple[float, float]:
    """
    Fit model to training data and predict the next point.
    
    Args:
        x_train: Training x values
        y_train: Training y values
        x_new: New x value to predict
        degree: Polynomial degree
        
    Returns:
        Tuple of (prediction, estimated_sigma)
        
    Example:
        >>> x_train = np.array([0, 1, 2, 3])
        >>> y_train = np.array([0, 1, 4, 9])  # Roughly quadratic
        >>> pred, sigma = predict_next_point(x_train, y_train, 4, 2)
        >>> print(f"Predicted y(4) = {pred:.2f} ± {sigma:.2f}")
    """
    # TODO: Implement prediction
    # Steps:
    #   1. Fit polynomial to training data
    #   2. Compute training residuals to estimate sigma
    #   3. Predict y value for x_new
    
    pass  # Your code here


def code_length_for_point(
    y_actual: float,
    y_predicted: float,
    sigma: float
) -> float:
    """
    Compute the code length (bits) for encoding one observation.
    
    Using Gaussian coding:
        L = 0.5 * log2(2π σ²) + (y - ŷ)² / (2σ² ln(2))
    
    Intuition: The further y_actual is from y_predicted,
    the more "surprising" it is, and the more bits needed.
    
    Args:
        y_actual: The actual observed value
        y_predicted: The model's prediction
        sigma: Estimated standard deviation
        
    Returns:
        Code length in bits
        
    Example:
        >>> # Small error = short code
        >>> code_length_for_point(10.0, 10.1, 1.0)  # Low surprise
        >>> # Large error = long code
        >>> code_length_for_point(10.0, 15.0, 1.0)  # High surprise
    """
    # TODO: Implement Gaussian code length
    # Hint: sigma should be at least 1e-10 to avoid division by zero
    
    pass  # Your code here


def prequential_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    Compute the Prequential MDL score for polynomial regression.
    
    Algorithm:
    1. For i = start_index to n:
       a. Fit model on points 0..i-1
       b. Predict point i
       c. Compute code length for actual vs predicted
       d. Add to total
    
    Args:
        x: Input values (shape: [n])
        y: Target values (shape: [n])
        degree: Polynomial degree
        start_index: Where to start predictions (default: degree + 2)
        
    Returns:
        Total prequential code length in bits
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> score_1 = prequential_mdl(x, y, 1)  # Underfit
        >>> score_2 = prequential_mdl(x, y, 2)  # Just right
        >>> score_5 = prequential_mdl(x, y, 5)  # Overfit
        >>> # score_2 should be lowest
    """
    # TODO: Implement prequential MDL
    # Steps:
    #   1. Determine start_index (need at least degree+2 points)
    #   2. Loop from start_index to n
    #   3. For each point: fit, predict, compute code length
    #   4. Sum all code lengths
    
    pass  # Your code here


def prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Tuple[int, dict]:
    """
    Select the best polynomial degree using Prequential MDL.
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to consider
        
    Returns:
        Tuple of (best_degree, scores_dict)
        
    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> best, scores = prequential_model_selection(x, y)
        >>> print(f"Best degree: {best}")
    """
    # TODO: Compute prequential MDL for each degree, return best
    
    pass  # Your code here


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_implementation():
    """Test your prequential MDL implementation."""
    
    print("Testing Prequential MDL Implementation")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: predict_next_point
    print("\n1. Testing predict_next_point()...")
    try:
        x_train = np.array([0, 1, 2, 3, 4])
        y_train = np.array([0, 1, 4, 9, 16])  # Perfect squares
        
        pred, sigma = predict_next_point(x_train, y_train, 5, 2)
        
        # Should predict close to 25
        assert 20 < pred < 30, f"Expected ~25, got {pred}"
        assert sigma >= 0, "Sigma should be non-negative"
        
        print(f"   Predicting y(5) from [0,1,4,9,16]")
        print(f"   Prediction: {pred:.2f}, Sigma: {sigma:.4f}")
        print("   ✓ PASSED")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 2: code_length_for_point
    print("\n2. Testing code_length_for_point()...")
    try:
        sigma = 1.0
        
        # Small error
        small_code = code_length_for_point(10.0, 10.1, sigma)
        # Large error
        large_code = code_length_for_point(10.0, 15.0, sigma)
        
        assert large_code > small_code, "Larger errors should have longer codes"
        assert small_code > 0, "Code length should be positive"
        
        print(f"   Error = 0.1: {small_code:.2f} bits")
        print(f"   Error = 5.0: {large_code:.2f} bits")
        print("   ✓ PASSED")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 3: prequential_mdl
    print("\n3. Testing prequential_mdl()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # Quadratic
        
        score_1 = prequential_mdl(x, y, 1)
        score_2 = prequential_mdl(x, y, 2)
        score_5 = prequential_mdl(x, y, 5)
        
        print(f"   Prequential MDL (degree 1): {score_1:.2f} bits")
        print(f"   Prequential MDL (degree 2): {score_2:.2f} bits")
        print(f"   Prequential MDL (degree 5): {score_5:.2f} bits")
        
        # Degree 2 should have lowest (best) score
        if score_2 < score_1 and score_2 < score_5:
            print("   ✓ PASSED - Degree 2 has lowest score!")
        else:
            print("   ⚠ Score ordering unexpected (might be noise)")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 4: prequential_model_selection
    print("\n4. Testing prequential_model_selection()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
        
        best_degree, scores = prequential_model_selection(x, y, max_degree=6)
        
        print(f"   Scores: {[(d, f'{s:.0f}') for d, s in scores.items()]}")
        print(f"   Selected: degree {best_degree}")
        print(f"   True: degree 2")
        
        if best_degree == 2:
            print("   ✓ PASSED - Found true degree!")
        else:
            print(f"   ⚠ Selected {best_degree} (may vary with noise)")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # Test 5: Comparison with Two-Part Code
    print("\n5. Comparing with Two-Part Code...")
    try:
        # Both methods should generally agree on well-structured data
        x = np.linspace(0, 10, 100)
        y = 5 * np.sin(x) + np.random.randn(100) * 0.5
        
        preq_best, _ = prequential_model_selection(x, y, max_degree=8)
        
        print(f"   Prequential selects: degree {preq_best}")
        print("   (Compare this with two-part code in exercise 1)")
        print("   ✓ Test complete")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    test_implementation()
