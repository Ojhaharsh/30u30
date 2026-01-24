"""
Day 5: MDL (Minimum Description Length) Principle - Implementation
=====================================================================

A complete implementation of MDL-based model selection, covering:
- Two-Part Codes (crude MDL)
- Prequential (Predictive) MDL
- Normalized Maximum Likelihood (NML)
- Practical model selection for polynomials and linear models

Paper: "A Tutorial Introduction to the Minimum Description Length Principle"
        by Peter Grünwald (2004)

Think of this as: A library for finding the SHORTEST description of your data.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# SECTION 1: CORE MDL COMPUTATIONS
# =============================================================================

def log2(x: float) -> float:
    """Safe log base 2, returns 0 for x <= 0."""
    if x <= 0:
        return 0.0
    return np.log2(x)


def universal_code_integers(n: int) -> float:
    """
    Universal prior code length for positive integers.
    
    This is the "fair" way to encode integers when you don't know
    their range in advance. Used for encoding model parameters.
    
    L(n) = log*(n) + log(c₀)
    where log*(n) = log(n) + log(log(n)) + ... (only positive terms)
    
    The Spy Analogy:
    ================
    Imagine telling your spy friend a number without knowing the range.
    You can't just use 4 bits (that assumes max 15).
    
    Universal code says: "First tell me how many digits, then the number."
    This works for ANY number, big or small.
    
    Args:
        n: Positive integer to encode
        
    Returns:
        Code length in bits
    """
    if n <= 0:
        return float('inf')
    
    # Log-star: sum of iterated logs until negative
    c0 = 2.865064  # Normalization constant for valid prefix code
    length = log2(c0)
    
    current = float(n)
    while current > 1:
        length += log2(current)
        current = log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def precision_code_length(precision_bits: int = 32) -> float:
    """
    Code length for specifying precision of real numbers.
    
    When encoding real numbers, we must specify the precision.
    Higher precision = more bits needed.
    
    Args:
        precision_bits: Bits per floating-point number
        
    Returns:
        Length of precision specification in bits
    """
    return universal_code_integers(precision_bits)


def real_number_code_length(value: float, precision_bits: int = 32) -> float:
    """
    Code length for encoding a single real number.
    
    We need to encode:
    1. The sign (1 bit)
    2. The exponent (encoded with universal integer code)
    3. The mantissa (precision_bits)
    
    Args:
        value: Real number to encode
        precision_bits: Bits for mantissa
        
    Returns:
        Total code length in bits
    """
    if value == 0:
        return 1  # Just encode "zero" flag
    
    sign_bit = 1
    
    # Exponent encoding
    if abs(value) >= 1:
        exponent = int(np.floor(np.log2(abs(value)))) + 1
    else:
        exponent = -int(np.ceil(-np.log2(abs(value))))
    
    exponent_length = universal_code_integers(abs(exponent) + 1) + 1  # +1 for sign
    
    return sign_bit + exponent_length + precision_bits


# =============================================================================
# SECTION 2: TWO-PART CODE MDL
# =============================================================================

def two_part_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    Compute Two-Part MDL score for polynomial regression.
    
    Two-Part Code = L(Model) + L(Data|Model)
    
    The Spy Analogy (from README):
    ==============================
    You're a spy sending temperature readings to headquarters.
    
    L(Model): Cost of sending the polynomial formula
              "Use: T = 23.5 + 0.02*hour - 0.001*hour²"
              
    L(Data|Model): Cost of sending the deviations
                   "Actual readings differ from formula by: +0.1, -0.3, ..."
    
    The best polynomial minimizes the TOTAL message length.
    
    Args:
        x: Input values (shape: [n])
        y: Target values (shape: [n])
        degree: Polynomial degree (0 = constant, 1 = line, etc.)
        precision_bits: Bits per floating-point coefficient
        
    Returns:
        Tuple of (total_mdl, model_cost, data_cost)
    """
    n = len(x)
    
    # Fit polynomial
    try:
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        predictions = poly(x)
        residuals = y - predictions
    except np.linalg.LinAlgError:
        return float('inf'), float('inf'), 0.0
    
    # =========================================================================
    # PART 1: Model Description Length L(H)
    # =========================================================================
    
    # Cost to encode degree (which polynomial?)
    degree_cost = universal_code_integers(degree + 1)
    
    # Cost to encode coefficients
    num_params = degree + 1
    param_cost = num_params * (precision_bits + 1)  # +1 for sign
    
    # Cost to encode precision
    precision_cost = precision_code_length(precision_bits)
    
    model_cost = degree_cost + param_cost + precision_cost
    
    # =========================================================================
    # PART 2: Data Description Length L(D|H)
    # =========================================================================
    
    # We encode residuals using a Gaussian model
    # Under optimal Gaussian coding, each residual costs:
    # log(σ√(2πe)) + (residual²)/(2σ²ln2) bits
    
    sigma = np.std(residuals) if np.std(residuals) > 0 else 1e-10
    
    # Simplified: Use log-likelihood conversion
    # bits = -log₂(P(data|model))
    
    if sigma > 0:
        # Gaussian log-likelihood
        nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
              np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
        
        # Add cost to encode sigma
        sigma_cost = real_number_code_length(sigma, precision_bits)
        data_cost = nll + sigma_cost
    else:
        data_cost = 0  # Perfect fit (rare)
    
    total_mdl = model_cost + data_cost
    
    return total_mdl, model_cost, data_cost


def select_polynomial_degree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    precision_bits: int = 32
) -> Dict[str, Any]:
    """
    Select optimal polynomial degree using Two-Part MDL.
    
    This is the practical "just use this" function for polynomial model selection.
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(100) * 0.5  # True: degree 2
    >>> result = select_polynomial_degree(x, y)
    >>> print(f"Best degree: {result['best_degree']}")  # Should be 2
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to consider
        precision_bits: Bits per coefficient
        
    Returns:
        Dictionary with:
        - best_degree: Optimal polynomial degree
        - scores: Dict mapping degree -> MDL score
        - coefficients: Fitted coefficients for best model
        - model_cost: L(H) for best model
        - data_cost: L(D|H) for best model
    """
    scores = {}
    details = {}
    
    for degree in range(max_degree + 1):
        total, model, data = two_part_mdl_polynomial(x, y, degree, precision_bits)
        scores[degree] = total
        details[degree] = {'model_cost': model, 'data_cost': data}
    
    best_degree = min(scores, key=scores.get)
    
    # Get coefficients for best model
    coefficients = np.polyfit(x, y, best_degree)
    
    return {
        'best_degree': best_degree,
        'scores': scores,
        'details': details,
        'coefficients': coefficients,
        'model_cost': details[best_degree]['model_cost'],
        'data_cost': details[best_degree]['data_cost'],
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# SECTION 3: PREQUENTIAL (PREDICTIVE) MDL
# =============================================================================

def prequential_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    Compute Prequential MDL score for polynomial regression.
    
    Prequential = "Predict Each, Then Learn From It"
    
    The Weather Forecaster Analogy:
    ================================
    Imagine a weather forecaster who:
    1. Predicts tomorrow's temperature (BEFORE seeing it)
    2. Records how surprised they are by the actual value
    3. Updates their model based on the new observation
    4. Repeat for each day
    
    Total "surprise" across all days = Prequential MDL score
    
    This is MORE ROBUST than two-part code because:
    - No need to specify parameter precision
    - Naturally handles model complexity
    - Better for time series and sequential data
    
    Mathematical Form:
    ===================
    L_prequential = -Σ log₂ P(yᵢ | y₁, ..., yᵢ₋₁, model)
    
    Args:
        x: Input values (shape: [n])
        y: Target values (shape: [n])
        degree: Polynomial degree
        start_index: Where to start predictions (default: degree + 2)
        
    Returns:
        Prequential MDL score in bits
    """
    n = len(x)
    
    # Need at least degree+1 points to fit, plus one to predict
    if start_index is None:
        start_index = degree + 2
    
    if start_index >= n:
        return float('inf')
    
    total_code_length = 0.0
    
    for i in range(start_index, n):
        # Fit model on points 0..i-1
        x_train = x[:i]
        y_train = y[:i]
        
        try:
            coefficients = np.polyfit(x_train, y_train, degree)
            poly = np.poly1d(coefficients)
            
            # Predict point i
            y_pred = poly(x[i])
            
            # Estimate variance from training residuals
            train_pred = poly(x_train)
            residuals = y_train - train_pred
            sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
            
            # Code length for this prediction (Gaussian model)
            error = y[i] - y_pred
            code_length = 0.5 * np.log2(2 * np.pi * sigma**2) + \
                          (error**2) / (2 * sigma**2 * np.log(2))
            
            total_code_length += code_length
            
        except (np.linalg.LinAlgError, np.RankWarning):
            total_code_length += 100  # Penalty for numerical issues
    
    return total_code_length


def prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    Select best polynomial using Prequential MDL.
    
    Example:
    --------
    >>> x = np.linspace(0, 5, 50)
    >>> y = np.sin(x) + 0.1 * np.random.randn(50)
    >>> result = prequential_model_selection(x, y)
    >>> print(f"Best degree: {result['best_degree']}")
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to try
        
    Returns:
        Dictionary with best_degree, scores, etc.
    """
    scores = {}
    
    for degree in range(max_degree + 1):
        scores[degree] = prequential_mdl_polynomial(x, y, degree)
    
    best_degree = min(scores, key=scores.get)
    coefficients = np.polyfit(x, y, best_degree)
    
    return {
        'best_degree': best_degree,
        'scores': scores,
        'coefficients': coefficients,
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# SECTION 4: NORMALIZED MAXIMUM LIKELIHOOD (NML)
# =============================================================================

def nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = (-100, 100),
    num_samples: int = 1000
) -> float:
    """
    Estimate NML (stochastic) complexity for polynomial models.
    
    NML Complexity = How many different datasets can this model explain well?
    
    The Restaurant Analogy:
    =======================
    Model complexity is like a restaurant's menu:
    - Simple model (degree 1): Can only make "lines" (small menu)
    - Complex model (degree 9): Can make almost any shape (huge menu)
    
    NML says: "A bigger menu means more uncertainty about what you'll order."
    
    The complex model gets penalized because it could explain TOO many things,
    making it less impressive when it happens to fit your specific data.
    
    Mathematical Form:
    ==================
    COMP(M) = log Σ_{z∈Z} P(z | θ̂(z), M)
    
    where θ̂(z) is the ML estimate for each possible dataset z.
    
    This is estimated via Monte Carlo sampling.
    
    Args:
        x: Input values (fixed design points)
        degree: Polynomial degree
        y_range: Range of possible y values
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated NML complexity in bits
    """
    n = len(x)
    
    max_log_likelihood = []
    
    for _ in range(num_samples):
        # Sample random y values
        y_sample = np.random.uniform(y_range[0], y_range[1], n)
        
        # Fit polynomial (maximum likelihood)
        try:
            coefficients = np.polyfit(x, y_sample, min(degree, n - 1))
            poly = np.poly1d(coefficients)
            predictions = poly(x)
            
            # Compute maximum likelihood (under Gaussian noise)
            residuals = y_sample - predictions
            sigma_ml = np.sqrt(np.mean(residuals**2)) if np.any(residuals) else 1e-10
            sigma_ml = max(sigma_ml, 1e-10)
            
            # Log-likelihood at ML estimate
            log_lik = -0.5 * n * np.log(2 * np.pi * sigma_ml**2) - \
                      np.sum(residuals**2) / (2 * sigma_ml**2)
            
            max_log_likelihood.append(log_lik)
            
        except:
            continue
    
    if not max_log_likelihood:
        return 0.0
    
    # NML complexity = log of average max-likelihood
    # log(1/N * Σ exp(log_lik)) ≈ logsumexp(log_liks) - log(N)
    max_ll = np.max(max_log_likelihood)
    complexity = max_ll + np.log(np.mean(np.exp(np.array(max_log_likelihood) - max_ll)))
    
    return complexity / np.log(2)  # Convert to bits


def nml_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """
    Compute NML-based MDL score for polynomial regression.
    
    NML MDL = -log P(data | ML params) + COMP(model)
    
    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree
        y_range: Range for complexity estimation (auto-detected if None)
        
    Returns:
        Tuple of (total_nml, fit_term, complexity_term)
    """
    n = len(x)
    
    if y_range is None:
        margin = 2 * (np.max(y) - np.min(y))
        y_range = (np.min(y) - margin, np.max(y) + margin)
    
    # Fit polynomial
    try:
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        predictions = poly(x)
        residuals = y - predictions
    except:
        return float('inf'), float('inf'), 0.0
    
    # Maximum likelihood fit (negative log-likelihood in bits)
    sigma_ml = np.sqrt(np.mean(residuals**2)) if np.any(residuals) else 1e-10
    sigma_ml = max(sigma_ml, 1e-10)
    
    fit_term = (0.5 * n * np.log2(2 * np.pi * sigma_ml**2) + 
                np.sum(residuals**2) / (2 * sigma_ml**2 * np.log(2)))
    
    # Complexity penalty
    complexity = nml_complexity_polynomial(x, degree, y_range, num_samples=500)
    
    total = fit_term + complexity
    
    return total, fit_term, complexity


# =============================================================================
# SECTION 5: MODEL CLASSES FOR GENERAL MDL
# =============================================================================

@dataclass
class MDLModel(ABC):
    """
    Abstract base class for MDL-compatible models.
    
    To create your own MDL model:
    1. Inherit from MDLModel
    2. Implement fit(), predict(), model_code_length(), and data_code_length()
    
    Example:
    --------
    class MyCustomModel(MDLModel):
        def fit(self, x, y): ...
        def predict(self, x): ...
        def model_code_length(self): ...
        def data_code_length(self, x, y): ...
    """
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def model_code_length(self) -> float:
        """Return L(H) - bits to describe the model."""
        pass
    
    @abstractmethod
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return L(D|H) - bits to describe data given model."""
        pass
    
    def total_mdl(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return total MDL = L(H) + L(D|H)."""
        return self.model_code_length() + self.data_code_length(x, y)


class PolynomialMDL(MDLModel):
    """
    Polynomial model with MDL scoring.
    
    Example:
    --------
    >>> model = PolynomialMDL(degree=2)
    >>> model.fit(x, y)
    >>> score = model.total_mdl(x, y)
    >>> predictions = model.predict(x_test)
    """
    
    def __init__(self, degree: int, precision_bits: int = 32):
        self.degree = degree
        self.precision_bits = precision_bits
        self.coefficients = None
        self.sigma = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit polynomial to data."""
        self.coefficients = np.polyfit(x, y, self.degree)
        self.poly = np.poly1d(self.coefficients)
        residuals = y - self.poly(x)
        self.sigma = np.std(residuals) if np.std(residuals) > 0 else 1e-10
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values."""
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.poly(x)
    
    def model_code_length(self) -> float:
        """Bits to describe polynomial model."""
        if self.coefficients is None:
            return float('inf')
        
        # Degree specification
        degree_cost = universal_code_integers(self.degree + 1)
        
        # Coefficients
        num_params = len(self.coefficients)
        param_cost = num_params * (self.precision_bits + 1)
        
        return degree_cost + param_cost
    
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bits to describe residuals."""
        if self.coefficients is None:
            self.fit(x, y)
        
        predictions = self.predict(x)
        residuals = y - predictions
        n = len(y)
        
        # Gaussian coding
        nll = 0.5 * n * np.log2(2 * np.pi * self.sigma**2) + \
              np.sum(residuals**2) / (2 * self.sigma**2 * np.log(2))
        
        # Add sigma encoding cost
        sigma_cost = real_number_code_length(self.sigma, self.precision_bits)
        
        return nll + sigma_cost


class GaussianMDL(MDLModel):
    """
    Simple Gaussian model with MDL scoring.
    
    Models data as y ~ N(μ, σ²).
    
    Useful as a baseline - if a more complex model can't beat this,
    there's no significant structure in the data.
    """
    
    def __init__(self, precision_bits: int = 32):
        self.precision_bits = precision_bits
        self.mean = None
        self.std = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit Gaussian to y values (ignores x)."""
        self.mean = np.mean(y)
        self.std = np.std(y) if np.std(y) > 0 else 1e-10
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict mean everywhere."""
        return np.full(len(x), self.mean)
    
    def model_code_length(self) -> float:
        """Bits to describe μ and σ."""
        if self.mean is None:
            return float('inf')
        
        return (real_number_code_length(self.mean, self.precision_bits) +
                real_number_code_length(self.std, self.precision_bits))
    
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bits to describe deviations from mean."""
        if self.mean is None:
            self.fit(x, y)
        
        n = len(y)
        nll = 0.5 * n * np.log2(2 * np.pi * self.std**2) + \
              np.sum((y - self.mean)**2) / (2 * self.std**2 * np.log(2))
        
        return nll


# =============================================================================
# SECTION 6: COMPARISON WITH AIC/BIC
# =============================================================================

def aic_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> float:
    """
    Compute AIC (Akaike Information Criterion) score.
    
    AIC = 2k - 2ln(L)
    
    where k = number of parameters, L = maximum likelihood.
    
    Lower is better.
    """
    n = len(x)
    k = degree + 2  # Polynomial coeffs + variance
    
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        rss = np.sum(residuals**2)
        
        # Log-likelihood (Gaussian)
        sigma_ml = np.sqrt(rss / n)
        if sigma_ml <= 0:
            sigma_ml = 1e-10
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
        
        return 2 * k - 2 * log_likelihood
        
    except:
        return float('inf')


def bic_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> float:
    """
    Compute BIC (Bayesian Information Criterion) score.
    
    BIC = k*ln(n) - 2ln(L)
    
    where k = number of parameters, n = sample size, L = maximum likelihood.
    
    Lower is better. BIC has a stronger penalty than AIC for large n.
    """
    n = len(x)
    k = degree + 2  # Polynomial coeffs + variance
    
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        rss = np.sum(residuals**2)
        
        # Log-likelihood (Gaussian)
        sigma_ml = np.sqrt(rss / n)
        if sigma_ml <= 0:
            sigma_ml = 1e-10
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
        
        return k * np.log(n) - 2 * log_likelihood
        
    except:
        return float('inf')


def compare_mdl_aic_bic(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    Compare MDL, AIC, and BIC for polynomial model selection.
    
    This is the "showdown" function to see how the criteria differ.
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2 + 3*x - 0.5*x**2 + np.random.randn(50) * 2  # True degree: 2
    >>> results = compare_mdl_aic_bic(x, y)
    >>> print(f"MDL picks: {results['mdl_best']}")
    >>> print(f"AIC picks: {results['aic_best']}")
    >>> print(f"BIC picks: {results['bic_best']}")
    
    Returns:
        Dictionary with scores and best choices for each criterion.
    """
    mdl_scores = {}
    aic_scores = {}
    bic_scores = {}
    
    for degree in range(max_degree + 1):
        mdl_scores[degree] = two_part_mdl_polynomial(x, y, degree)[0]
        aic_scores[degree] = aic_score(x, y, degree)
        bic_scores[degree] = bic_score(x, y, degree)
    
    return {
        'mdl_scores': mdl_scores,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores,
        'mdl_best': min(mdl_scores, key=mdl_scores.get),
        'aic_best': min(aic_scores, key=aic_scores.get),
        'bic_best': min(bic_scores, key=bic_scores.get)
    }


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS
# =============================================================================

def compression_ratio(
    data: np.ndarray,
    model_mdl: float
) -> float:
    """
    Compute compression ratio achieved by MDL model.
    
    Ratio > 1: Model compresses data (good!)
    Ratio < 1: Model expands data (bad!)
    Ratio = 1: No compression (data might be random)
    
    Args:
        data: Original data array
        model_mdl: MDL score of the model
        
    Returns:
        Compression ratio
    """
    # Raw data cost: float32 per element
    raw_bits = len(data) * 32
    
    return raw_bits / model_mdl if model_mdl > 0 else 0


def mdl_model_probability(
    mdl_scores: Dict[int, float]
) -> Dict[int, float]:
    """
    Convert MDL scores to approximate posterior probabilities.
    
    Using: P(model) ∝ 2^(-MDL score)
    
    This gives an intuitive sense of how much better one model is.
    
    Args:
        mdl_scores: Dict mapping model ID -> MDL score
        
    Returns:
        Dict mapping model ID -> probability
    """
    # Normalize to prevent overflow
    min_score = min(mdl_scores.values())
    normalized = {k: -(v - min_score) for k, v in mdl_scores.items()}
    
    # Convert to probabilities
    powers = {k: 2**v for k, v in normalized.items()}
    total = sum(powers.values())
    
    return {k: v / total for k, v in powers.items()}


def generate_test_data(
    true_degree: int = 2,
    n_points: int = 50,
    noise_std: float = 1.0,
    x_range: Tuple[float, float] = (0, 10),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic polynomial data for testing.
    
    Args:
        true_degree: True polynomial degree
        n_points: Number of data points
        noise_std: Standard deviation of Gaussian noise
        x_range: Range of x values
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (x, y_noisy, true_coefficients)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Generate random coefficients
    true_coeffs = np.random.randn(true_degree + 1) * 2
    poly = np.poly1d(true_coeffs)
    
    y_true = poly(x)
    y_noisy = y_true + np.random.randn(n_points) * noise_std
    
    return x, y_noisy, true_coeffs


# =============================================================================
# SECTION 8: MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrate MDL model selection on synthetic data.
    """
    print("=" * 70)
    print("MDL (Minimum Description Length) Principle - Implementation Demo")
    print("=" * 70)
    print()
    
    # Generate test data
    np.random.seed(42)
    true_degree = 2
    x, y, true_coeffs = generate_test_data(
        true_degree=true_degree,
        n_points=50,
        noise_std=1.5,
        seed=42
    )
    
    print(f"Generated data with TRUE degree = {true_degree}")
    print(f"True coefficients: {true_coeffs}")
    print(f"Number of points: {len(x)}")
    print()
    
    # =========================================================================
    # Two-Part Code MDL
    # =========================================================================
    print("-" * 50)
    print("METHOD 1: Two-Part Code MDL")
    print("-" * 50)
    
    result = select_polynomial_degree(x, y, max_degree=10)
    
    print(f"\nMDL Scores by Degree:")
    for deg in range(6):
        score = result['scores'][deg]
        marker = " <-- BEST" if deg == result['best_degree'] else ""
        print(f"  Degree {deg}: {score:.2f} bits{marker}")
    
    print(f"\nTwo-Part MDL selects: Degree {result['best_degree']}")
    print(f"  Model cost: {result['model_cost']:.2f} bits")
    print(f"  Data cost: {result['data_cost']:.2f} bits")
    print(f"  Total: {result['total_mdl']:.2f} bits")
    
    # =========================================================================
    # Prequential MDL
    # =========================================================================
    print()
    print("-" * 50)
    print("METHOD 2: Prequential MDL")
    print("-" * 50)
    
    preq_result = prequential_model_selection(x, y, max_degree=10)
    
    print(f"\nPrequential MDL Scores:")
    for deg in range(6):
        score = preq_result['scores'][deg]
        marker = " <-- BEST" if deg == preq_result['best_degree'] else ""
        print(f"  Degree {deg}: {score:.2f} bits{marker}")
    
    print(f"\nPrequential MDL selects: Degree {preq_result['best_degree']}")
    
    # =========================================================================
    # Comparison with AIC and BIC
    # =========================================================================
    print()
    print("-" * 50)
    print("COMPARISON: MDL vs AIC vs BIC")
    print("-" * 50)
    
    comparison = compare_mdl_aic_bic(x, y, max_degree=10)
    
    print(f"\nBest degree selected by each criterion:")
    print(f"  MDL: Degree {comparison['mdl_best']}")
    print(f"  AIC: Degree {comparison['aic_best']}")
    print(f"  BIC: Degree {comparison['bic_best']}")
    print(f"  True: Degree {true_degree}")
    
    # =========================================================================
    # Model Probabilities
    # =========================================================================
    print()
    print("-" * 50)
    print("MDL MODEL PROBABILITIES")
    print("-" * 50)
    
    probs = mdl_model_probability(result['scores'])
    
    print("\nPosterior probability of each degree:")
    for deg in range(6):
        bar = "█" * int(probs[deg] * 50)
        print(f"  Degree {deg}: {probs[deg]:.4f} {bar}")
    
    # =========================================================================
    # Compression Analysis
    # =========================================================================
    print()
    print("-" * 50)
    print("COMPRESSION ANALYSIS")
    print("-" * 50)
    
    raw_bits = len(y) * 32
    best_mdl = result['total_mdl']
    ratio = compression_ratio(y, best_mdl)
    
    print(f"\nRaw data size: {raw_bits} bits ({len(y)} × 32-bit floats)")
    print(f"MDL compressed: {best_mdl:.0f} bits")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"\n{'✓ Found compressible structure!' if ratio > 1 else '✗ Data appears random.'}")
    
    print()
    print("=" * 70)
    print("Demo complete! Run visualization.py for plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
