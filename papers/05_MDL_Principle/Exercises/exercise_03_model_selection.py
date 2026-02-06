"""
Exercise 3: Practical Model Selection with MDL
===============================================

In this exercise, you'll build a complete model selection system using MDL.

The goal: Given noisy data, find the TRUE underlying pattern.

Real-world applications:
- Choosing neural network architecture
- Selecting features for regression
- Determining the number of clusters
- Picking the best grammar for language modeling

The Detective Analogy:
---------------------
You're a detective with multiple theories about a crime.
Each theory has a "cost" (how complex it is) and
an "explanation power" (how well it fits the evidence).

MDL finds the theory with the best balance.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class MDLModelSelector:
    """
    A class for MDL-based polynomial model selection.
    
    This encapsulates the full workflow:
    1. Compute MDL scores for different models
    2. Select the best model
    3. Analyze the trade-offs
    4. Make predictions
    """
    
    def __init__(self, max_degree: int = 10, precision_bits: int = 32):
        """
        Initialize the model selector.
        
        Args:
            max_degree: Maximum polynomial degree to consider
            precision_bits: Bits for encoding coefficients
        """
        self.max_degree = max_degree
        self.precision_bits = precision_bits
        self.scores: Dict[int, float] = {}
        self.model_costs: Dict[int, float] = {}
        self.data_costs: Dict[int, float] = {}
        self.best_degree: Optional[int] = None
        self.best_coefficients: Optional[np.ndarray] = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'MDLModelSelector':
        """
        Fit the model selector to data.
        
        Computes MDL scores for all degrees from 0 to max_degree,
        then selects the degree with minimum total MDL.
        
        Args:
            x: Input values
            y: Target values
            
        Returns:
            self (for method chaining)
            
        Example:
            >>> selector = MDLModelSelector(max_degree=10)
            >>> selector.fit(x, y)
            >>> print(f"Best degree: {selector.best_degree}")
        """
        # TODO: Implement the fitting process
        # Steps:
        #   1. Loop through degrees 0 to max_degree
        #   2. For each degree, compute MDL (model cost + data cost)
        #   3. Store scores in self.scores, etc.
        #   4. Find best_degree (minimum total MDL)
        #   5. Fit final model and store coefficients
        
        pass  # Your code here
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the selected model.
        
        Args:
            x: Input values to predict
            
        Returns:
            Predicted y values
            
        Raises:
            ValueError: If fit() hasn't been called
        """
        # TODO: Implement prediction
        
        pass  # Your code here
    
    def get_compression_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the compression ratio achieved.
        
        Compression ratio = Raw data bits / MDL bits
        
        A ratio > 1 means we found compressible structure!
        
        Args:
            x: Input values
            y: Target values
            
        Returns:
            Compression ratio
        """
        # TODO: Implement compression ratio calculation
        # Raw bits: len(y) * 32 (assuming 32-bit floats)
        # MDL bits: self.scores[self.best_degree]
        
        pass  # Your code here
    
    def get_model_probability(self, degree: int) -> float:
        """
        Get the approximate posterior probability for a given degree.
        
        Using: P(model) ∝ 2^(-MDL score)
        
        Args:
            degree: The polynomial degree
            
        Returns:
            Approximate probability (sums to 1 across all degrees)
        """
        # TODO: Convert MDL scores to probabilities
        # Hint: Normalize by subtracting min score first to prevent overflow
        
        pass  # Your code here
    
    def summary(self) -> str:
        """
        Return a summary of the model selection results.
        """
        if self.best_degree is None:
            return "Model not fitted. Call fit() first."
        
        lines = [
            "MDL Model Selection Summary",
            "=" * 40,
            f"Best degree: {self.best_degree}",
            f"Model cost: {self.model_costs.get(self.best_degree, 0):.1f} bits",
            f"Data cost: {self.data_costs.get(self.best_degree, 0):.1f} bits",
            f"Total MDL: {self.scores.get(self.best_degree, 0):.1f} bits",
            "",
            "All scores:",
        ]
        
        for deg in sorted(self.scores.keys()):
            marker = " ← BEST" if deg == self.best_degree else ""
            lines.append(f"  Degree {deg}: {self.scores[deg]:.1f}{marker}")
        
        return "\n".join(lines)


def compare_with_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 5
) -> Dict[str, int]:
    """
    Compare MDL selection with k-fold cross-validation.
    
    This shows that MDL often agrees with CV but is computationally cheaper!
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to try
        n_folds: Number of CV folds
        
    Returns:
        Dictionary with 'mdl_best' and 'cv_best' degrees
        
    Example:
        >>> result = compare_with_cross_validation(x, y)
        >>> print(f"MDL: {result['mdl_best']}, CV: {result['cv_best']}")
    """
    # TODO: Implement comparison
    # Steps:
    #   1. Use MDLModelSelector to find best degree
    #   2. Implement k-fold CV to find best degree
    #   3. Return both
    
    pass  # Your code here


def monte_carlo_evaluation(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """
    Evaluate MDL model selection over many random trials.
    
    This tests how reliably MDL finds the true model.
    
    Args:
        true_degree: The actual polynomial degree
        n_samples: Number of data points per trial
        noise_std: Standard deviation of noise
        n_trials: Number of random trials
        
    Returns:
        Dictionary with accuracy statistics
    """
    # TODO: Implement Monte Carlo evaluation
    # Steps:
    #   1. For each trial:
    #      a. Generate random polynomial with true_degree
    #      b. Add noise
    #      c. Use MDLModelSelector to find best degree
    #      d. Check if it matches true_degree
    #   2. Compute accuracy
    
    pass  # Your code here


# =============================================================================
# HELPER FUNCTIONS (Implement these for the class to work)
# =============================================================================

def universal_code_length(n: int) -> float:
    """Universal code length for positive integers."""
    # TODO: Copy from exercise 1 or implement
    pass


def compute_mdl(x: np.ndarray, y: np.ndarray, degree: int, 
                precision: int = 32) -> Tuple[float, float, float]:
    """
    Compute two-part MDL.
    
    Returns: (total, model_cost, data_cost)
    """
    # TODO: Copy from exercise 1 or implement
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_implementation():
    """Test the MDL model selector."""
    
    print("Testing MDL Model Selection System")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Generate test data
    x = np.linspace(0, 10, 50)
    true_degree = 2
    true_coeffs = [-0.5, 2, 3]  # -0.5x² + 2x + 3
    y_true = 3 + 2*x - 0.5*x**2
    y = y_true + np.random.randn(50) * 1.5
    
    print(f"\nTest data: n={len(x)}, true degree={true_degree}")
    
    # Test 1: Basic fitting
    print("\n1. Testing MDLModelSelector.fit()...")
    try:
        selector = MDLModelSelector(max_degree=8)
        selector.fit(x, y)
        
        assert selector.best_degree is not None, "best_degree should be set"
        assert len(selector.scores) > 0, "scores should be populated"
        
        print(f"   Best degree selected: {selector.best_degree}")
        print(f"   True degree: {true_degree}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 2: Prediction
    print("\n2. Testing MDLModelSelector.predict()...")
    try:
        x_test = np.array([11, 12, 13])
        predictions = selector.predict(x_test)
        
        assert len(predictions) == len(x_test), "Wrong number of predictions"
        print(f"   Predictions for x=[11,12,13]: {predictions}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 3: Compression ratio
    print("\n3. Testing get_compression_ratio()...")
    try:
        ratio = selector.get_compression_ratio(x, y)
        
        assert ratio > 0, "Compression ratio should be positive"
        print(f"   Compression ratio: {ratio:.2f}x")
        
        if ratio > 1:
            print("   [ok] Data is compressible - structure found!")
        else:
            print("   [NOTE] No compression - data might be random")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 4: Model probabilities
    print("\n4. Testing get_model_probability()...")
    try:
        total_prob = sum(selector.get_model_probability(d) for d in range(9))
        
        assert abs(total_prob - 1.0) < 0.01, "Probabilities should sum to 1"
        
        print(f"   P(degree=2): {selector.get_model_probability(2):.4f}")
        print(f"   Sum of all probabilities: {total_prob:.4f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 5: Summary
    print("\n5. Testing summary()...")
    try:
        summary = selector.summary()
        assert "Best degree" in summary, "Summary should contain best degree"
        print("   Summary generated successfully")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # Test 6: Compare with CV (if implemented)
    print("\n6. Testing compare_with_cross_validation()...")
    try:
        result = compare_with_cross_validation(x, y, max_degree=6)
        print(f"   MDL selects: degree {result['mdl_best']}")
        print(f"   CV selects: degree {result['cv_best']}")
        
        if result['mdl_best'] == result['cv_best']:
            print("   [ok] MDL and CV agree!")
        else:
            print("   [NOTE] Methods disagree (happens sometimes)")
    except Exception as e:
        print(f"   [NOTE] Skipped (not implemented): {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nFull summary:")
    print(selector.summary())


if __name__ == "__main__":
    test_implementation()
