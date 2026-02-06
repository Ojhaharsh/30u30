"""
Solutions for Day 5: MDL (Minimum Description Length) Exercises
================================================================

WARNING: Try the exercises yourself first!
These solutions are for reference after you've attempted each problem.

Each solution includes:
1. Complete working code
2. Explanation of the approach
3. Key insights
"""

import numpy as np
from typing import Tuple, Dict, List


# =============================================================================
# EXERCISE 1 SOLUTIONS: Two-Part Code
# =============================================================================

def solution_universal_code_length(n: int) -> float:
    """
    Universal code for positive integers using log-star.
    
    L(n) = log₂(c₀) + log₂(n) + log₂(log₂(n)) + ...
    
    Only include positive terms in the sum.
    """
    if n <= 0:
        return float('inf')
    
    c0 = 2.865064  # Normalization constant
    length = np.log2(c0)
    
    current = float(n)
    while current > 1:
        length += np.log2(current)
        current = np.log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def solution_model_description_length(degree: int, precision_bits: int = 32) -> float:
    """
    Model cost = degree encoding + coefficient encoding.
    """
    # Encode degree
    degree_cost = solution_universal_code_length(degree + 1)
    
    # Encode coefficients (degree+1 of them)
    num_coefficients = degree + 1
    coefficient_cost = num_coefficients * (precision_bits + 1)  # +1 for sign
    
    return degree_cost + coefficient_cost


def solution_data_description_length(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> float:
    """
    Data cost = encode residuals using Gaussian model.
    """
    n = len(x)
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    predictions = poly(x)
    residuals = y - predictions
    
    # Estimate sigma
    sigma = np.std(residuals)
    sigma = max(sigma, 1e-10)  # Avoid division by zero
    
    # Gaussian negative log-likelihood (in bits)
    nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
          np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
    
    # Add cost to encode sigma
    sigma_cost = precision_bits + 10  # Simplified
    
    return nll + sigma_cost


def solution_two_part_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    Complete two-part MDL: L(H) + L(D|H).
    """
    model_cost = solution_model_description_length(degree, precision_bits)
    data_cost = solution_data_description_length(x, y, degree, precision_bits)
    total = model_cost + data_cost
    
    return total, model_cost, data_cost


# =============================================================================
# EXERCISE 2 SOLUTIONS: Prequential MDL
# =============================================================================

def solution_predict_next_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: float,
    degree: int
) -> Tuple[float, float]:
    """
    Fit model on training data and predict the next point.
    """
    # Fit polynomial
    coeffs = np.polyfit(x_train, y_train, degree)
    poly = np.poly1d(coeffs)
    
    # Estimate sigma from training residuals
    train_pred = poly(x_train)
    residuals = y_train - train_pred
    sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
    
    # Predict new point
    prediction = poly(x_new)
    
    return prediction, sigma


def solution_code_length_for_point(
    y_actual: float,
    y_predicted: float,
    sigma: float
) -> float:
    """
    Gaussian code length for one observation.
    """
    sigma = max(sigma, 1e-10)
    error = y_actual - y_predicted
    
    code_length = 0.5 * np.log2(2 * np.pi * sigma**2) + \
                  (error**2) / (2 * sigma**2 * np.log(2))
    
    return code_length


def solution_prequential_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    Prequential MDL: predict each point, sum the surprises.
    """
    n = len(x)
    
    # Need at least degree+1 points to fit, plus 1 to predict
    if start_index is None:
        start_index = degree + 2
    
    if start_index >= n:
        return float('inf')
    
    total_code_length = 0.0
    
    for i in range(start_index, n):
        # Training data: everything before point i
        x_train = x[:i]
        y_train = y[:i]
        
        # Predict point i
        prediction, sigma = solution_predict_next_point(
            x_train, y_train, x[i], degree
        )
        
        # Code length for this prediction
        code_length = solution_code_length_for_point(y[i], prediction, sigma)
        total_code_length += code_length
    
    return total_code_length


def solution_prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Tuple[int, dict]:
    """
    Select best degree using prequential MDL.
    """
    scores = {}
    
    for degree in range(max_degree + 1):
        scores[degree] = solution_prequential_mdl(x, y, degree)
    
    best_degree = min(scores, key=scores.get)
    
    return best_degree, scores


# =============================================================================
# EXERCISE 3 SOLUTIONS: Model Selection
# =============================================================================

class SolutionMDLModelSelector:
    """Complete implementation of MDL model selector."""
    
    def __init__(self, max_degree: int = 10, precision_bits: int = 32):
        self.max_degree = max_degree
        self.precision_bits = precision_bits
        self.scores = {}
        self.model_costs = {}
        self.data_costs = {}
        self.best_degree = None
        self.best_coefficients = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SolutionMDLModelSelector':
        """Fit model selector to data."""
        for deg in range(self.max_degree + 1):
            total, model, data = solution_two_part_mdl(
                x, y, deg, self.precision_bits
            )
            self.scores[deg] = total
            self.model_costs[deg] = model
            self.data_costs[deg] = data
        
        self.best_degree = min(self.scores, key=self.scores.get)
        self.best_coefficients = np.polyfit(x, y, self.best_degree)
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using selected model."""
        if self.best_coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.poly1d(self.best_coefficients)(x)
    
    def get_compression_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate compression ratio."""
        raw_bits = len(y) * 32
        mdl_bits = self.scores[self.best_degree]
        return raw_bits / mdl_bits if mdl_bits > 0 else 0
    
    def get_model_probability(self, degree: int) -> float:
        """Get approximate posterior probability."""
        min_score = min(self.scores.values())
        normalized = {k: -(v - min_score) for k, v in self.scores.items()}
        powers = {k: 2**v for k, v in normalized.items()}
        total = sum(powers.values())
        return powers[degree] / total if total > 0 else 0


def solution_compare_with_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 5
) -> Dict[str, int]:
    """Compare MDL with k-fold cross-validation."""
    # MDL selection
    selector = SolutionMDLModelSelector(max_degree=max_degree)
    selector.fit(x, y)
    mdl_best = selector.best_degree
    
    # K-fold CV
    n = len(x)
    fold_size = n // n_folds
    
    cv_errors = {}
    
    for deg in range(max_degree + 1):
        fold_errors = []
        
        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([np.arange(0, val_start), 
                                         np.arange(val_end, n)])
            
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            
            # Fit and evaluate
            coeffs = np.polyfit(x_train, y_train, deg)
            predictions = np.poly1d(coeffs)(x_val)
            mse = np.mean((y_val - predictions)**2)
            fold_errors.append(mse)
        
        cv_errors[deg] = np.mean(fold_errors)
    
    cv_best = min(cv_errors, key=cv_errors.get)
    
    return {'mdl_best': mdl_best, 'cv_best': cv_best}


# =============================================================================
# EXERCISE 4 SOLUTIONS: NML Complexity
# =============================================================================

def solution_logsumexp(log_values: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_values)
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


def solution_gaussian_log_likelihood(
    y: np.ndarray,
    predictions: np.ndarray,
    sigma: float = None
) -> float:
    """Gaussian log-likelihood (in nats, not bits)."""
    residuals = y - predictions
    n = len(y)
    
    if sigma is None:
        sigma = np.sqrt(np.mean(residuals**2))
    sigma = max(sigma, 1e-10)
    
    return -0.5 * n * np.log(2 * np.pi * sigma**2) - \
           np.sum(residuals**2) / (2 * sigma**2)


def solution_estimate_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = (-100, 100),
    n_samples: int = 1000
) -> float:
    """Estimate NML complexity via Monte Carlo."""
    n = len(x)
    max_log_likelihoods = []
    
    for _ in range(n_samples):
        # Sample random y values
        y_random = np.random.uniform(y_range[0], y_range[1], n)
        
        # Fit polynomial (ML estimate)
        try:
            coeffs = np.polyfit(x, y_random, min(degree, n - 1))
            predictions = np.poly1d(coeffs)(x)
            
            # ML sigma
            residuals = y_random - predictions
            sigma_ml = np.sqrt(np.mean(residuals**2))
            sigma_ml = max(sigma_ml, 1e-10)
            
            # Log-likelihood
            log_lik = solution_gaussian_log_likelihood(
                y_random, predictions, sigma_ml
            )
            max_log_likelihoods.append(log_lik)
        except:
            continue
    
    if not max_log_likelihoods:
        return 0.0
    
    # Complexity = log of average max-likelihood
    log_liks = np.array(max_log_likelihoods)
    complexity = solution_logsumexp(log_liks) - np.log(len(log_liks))
    
    return complexity / np.log(2)  # Convert to bits


def solution_compute_nml_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """Compute NML-based MDL score."""
    n = len(x)
    
    # Auto y_range
    if y_range is None:
        margin = 2 * (np.max(y) - np.min(y))
        y_range = (np.min(y) - margin, np.max(y) + margin)
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    predictions = np.poly1d(coeffs)(x)
    residuals = y - predictions
    
    # ML sigma
    sigma_ml = np.sqrt(np.mean(residuals**2))
    sigma_ml = max(sigma_ml, 1e-10)
    
    # Fit term (NLL in bits)
    fit_term = -solution_gaussian_log_likelihood(y, predictions, sigma_ml) / np.log(2)
    
    # Complexity
    complexity = solution_estimate_nml_complexity_polynomial(
        x, degree, y_range, n_samples=500
    )
    
    return fit_term + complexity, fit_term, complexity


# =============================================================================
# EXERCISE 5 SOLUTIONS: MDL vs AIC vs BIC
# =============================================================================

def solution_compute_aic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Compute AIC score."""
    n = len(x)
    k = degree + 2  # Coefficients + variance
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    
    return 2 * k - 2 * log_likelihood


def solution_compute_bic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Compute BIC score."""
    n = len(x)
    k = degree + 2
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    
    return k * np.log(n) - 2 * log_likelihood


def solution_select_with_all_criteria(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Dict]:
    """Run all three criteria."""
    results = {
        'mdl': {'scores': {}, 'best': None},
        'aic': {'scores': {}, 'best': None},
        'bic': {'scores': {}, 'best': None}
    }
    
    for deg in range(max_degree + 1):
        results['mdl']['scores'][deg] = solution_two_part_mdl(x, y, deg)[0]
        results['aic']['scores'][deg] = solution_compute_aic(x, y, deg)
        results['bic']['scores'][deg] = solution_compute_bic(x, y, deg)
    
    for method in ['mdl', 'aic', 'bic']:
        results[method]['best'] = min(
            results[method]['scores'],
            key=results[method]['scores'].get
        )
    
    return results


def solution_monte_carlo_comparison(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """Compare accuracy of all criteria."""
    x = np.linspace(0, 10, n_samples)
    
    correct = {'mdl': 0, 'aic': 0, 'bic': 0}
    
    for trial in range(n_trials):
        np.random.seed(trial)
        
        # Generate polynomial
        true_coeffs = np.random.randn(true_degree + 1) * 2
        y_true = np.poly1d(true_coeffs)(x)
        y = y_true + np.random.randn(n_samples) * noise_std
        
        # Run all criteria
        results = solution_select_with_all_criteria(x, y, max_degree=10)
        
        for method in ['mdl', 'aic', 'bic']:
            if results[method]['best'] == true_degree:
                correct[method] += 1
    
    return {
        'mdl_accuracy': correct['mdl'] / n_trials,
        'aic_accuracy': correct['aic'] / n_trials,
        'bic_accuracy': correct['bic'] / n_trials
    }


# =============================================================================
# MAIN: Verify solutions work
# =============================================================================

def verify_solutions():
    """Quick verification that solutions work."""
    print("Verifying Exercise Solutions")
    print("=" * 50)
    
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
    
    print("\n[ok] Exercise 1: Two-Part MDL")
    total, model, data = solution_two_part_mdl(x, y, 2)
    print(f"  MDL(degree=2) = {total:.1f} bits")
    
    print("\n[ok] Exercise 2: Prequential MDL")
    preq = solution_prequential_mdl(x, y, 2)
    print(f"  Prequential(degree=2) = {preq:.1f} bits")
    
    print("\n[ok] Exercise 3: Model Selection")
    selector = SolutionMDLModelSelector(max_degree=8)
    selector.fit(x, y)
    print(f"  Selected degree: {selector.best_degree}")
    
    print("\n[ok] Exercise 4: NML Complexity")
    complexity = solution_estimate_nml_complexity_polynomial(x, 2, n_samples=200)
    print(f"  COMP(degree=2) = {complexity:.1f} bits")
    
    print("\n[ok] Exercise 5: MDL vs AIC vs BIC")
    mc = solution_monte_carlo_comparison(n_trials=50)
    print(f"  MDL: {mc['mdl_accuracy']*100:.0f}%, "
          f"AIC: {mc['aic_accuracy']*100:.0f}%, "
          f"BIC: {mc['bic_accuracy']*100:.0f}%")
    
    print("\n" + "=" * 50)
    print("All solutions verified!")


if __name__ == "__main__":
    verify_solutions()
