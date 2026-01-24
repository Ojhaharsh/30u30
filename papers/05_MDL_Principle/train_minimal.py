"""
Day 5: MDL (Minimum Description Length) Principle - Minimal Training Script
=============================================================================

A complete, runnable demonstration of MDL-based model selection.
Run this to see MDL find the true polynomial degree in noisy data!

Usage:
    python train_minimal.py
    python train_minimal.py --true-degree 3 --noise 1.5 --samples 100

What this script does:
1. Generates synthetic polynomial data with known true degree
2. Applies MDL, AIC, and BIC to select the best model
3. Shows which method correctly identifies the true structure
4. Demonstrates the compression achieved

The key insight: MDL finds the model that COMPRESSES data most,
which usually means it found the TRUE underlying pattern.
"""

import numpy as np
import argparse
from typing import Tuple


# =============================================================================
# CORE MDL FUNCTIONS (Minimal version - no dependencies)
# =============================================================================

def universal_code_length(n: int) -> float:
    """Universal code for positive integers."""
    if n <= 0:
        return float('inf')
    
    c0 = 2.865064
    length = np.log2(c0)
    current = float(n)
    
    while current > 1:
        length += np.log2(current)
        current = np.log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def two_part_mdl(x: np.ndarray, y: np.ndarray, degree: int,
                  precision: int = 32) -> Tuple[float, float, float]:
    """
    Compute Two-Part MDL: L(Model) + L(Data|Model)
    
    Returns: (total, model_cost, data_cost)
    """
    n = len(x)
    
    # Fit polynomial
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
    except:
        return float('inf'), float('inf'), 0.0
    
    # Model cost: degree + coefficients
    model_cost = universal_code_length(degree + 1) + (degree + 1) * (precision + 1)
    
    # Data cost: encode residuals with Gaussian model
    sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
    nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
          np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
    
    # Add cost to specify sigma
    sigma_cost = precision + 10  # Simplified
    data_cost = nll + sigma_cost
    
    return model_cost + data_cost, model_cost, data_cost


def aic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Akaike Information Criterion."""
    n = len(x)
    k = degree + 2
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    return 2 * k - 2 * log_lik


def bic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Bayesian Information Criterion."""
    n = len(x)
    k = degree + 2
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    return k * np.log(n) - 2 * log_lik


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def run_experiment(
    true_degree: int = 2,
    noise_std: float = 1.5,
    n_samples: int = 50,
    max_degree: int = 10,
    seed: int = 42,
    verbose: bool = True
):
    """
    Run the MDL model selection experiment.
    
    Args:
        true_degree: The actual polynomial degree (what we're trying to recover)
        noise_std: Standard deviation of Gaussian noise
        n_samples: Number of data points
        max_degree: Maximum degree to consider
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with results
    """
    
    np.random.seed(seed)
    
    # =========================================================================
    # Generate synthetic data
    # =========================================================================
    
    x = np.linspace(0, 10, n_samples)
    
    # Random coefficients for true polynomial
    true_coeffs = np.random.randn(true_degree + 1) * 2
    poly_true = np.poly1d(true_coeffs)
    
    y_true = poly_true(x)
    y = y_true + np.random.randn(n_samples) * noise_std
    
    if verbose:
        print("\n" + "=" * 70)
        print("🎓 MDL (Minimum Description Length) - Model Selection Demo")
        print("=" * 70)
        print()
        print("📊 DATA GENERATION")
        print("-" * 40)
        print(f"  True polynomial degree: {true_degree}")
        print(f"  True coefficients: {np.round(true_coeffs, 2)}")
        print(f"  Number of samples: {n_samples}")
        print(f"  Noise std: {noise_std}")
        print(f"  Signal-to-Noise ratio: {np.std(y_true) / noise_std:.2f}")
    
    # =========================================================================
    # Compute scores for all degrees
    # =========================================================================
    
    mdl_scores = {}
    aic_scores = {}
    bic_scores = {}
    
    for deg in range(max_degree + 1):
        mdl_scores[deg] = two_part_mdl(x, y, deg)[0]
        aic_scores[deg] = aic(x, y, deg)
        bic_scores[deg] = bic(x, y, deg)
    
    # Find best for each criterion
    mdl_best = min(mdl_scores, key=mdl_scores.get)
    aic_best = min(aic_scores, key=aic_scores.get)
    bic_best = min(bic_scores, key=bic_scores.get)
    
    if verbose:
        print()
        print("📈 MDL SCORES BY DEGREE")
        print("-" * 40)
        print(f"{'Degree':<8} {'MDL (bits)':<15} {'AIC':<15} {'BIC':<15}")
        print("-" * 55)
        
        for deg in range(min(8, max_degree + 1)):
            mdl_marker = " ◀ MDL" if deg == mdl_best else ""
            aic_marker = " ◀ AIC" if deg == aic_best else ""
            bic_marker = " ◀ BIC" if deg == bic_best else ""
            
            print(f"{deg:<8} {mdl_scores[deg]:<15.2f} {aic_scores[deg]:<15.2f} "
                  f"{bic_scores[deg]:<15.2f}{mdl_marker}{aic_marker}{bic_marker}")
    
    # =========================================================================
    # Results summary
    # =========================================================================
    
    if verbose:
        print()
        print("🏆 MODEL SELECTION RESULTS")
        print("-" * 40)
        print(f"  True degree:  {true_degree}")
        print(f"  MDL selects:  {mdl_best} {'✓' if mdl_best == true_degree else '✗'}")
        print(f"  AIC selects:  {aic_best} {'✓' if aic_best == true_degree else '✗'}")
        print(f"  BIC selects:  {bic_best} {'✓' if bic_best == true_degree else '✗'}")
    
    # =========================================================================
    # Compression analysis
    # =========================================================================
    
    raw_bits = n_samples * 32  # 32 bits per float
    best_mdl_bits = mdl_scores[mdl_best]
    compression_ratio = raw_bits / best_mdl_bits
    
    if verbose:
        print()
        print("🗜️  COMPRESSION ANALYSIS")
        print("-" * 40)
        print(f"  Raw data size:      {raw_bits} bits")
        print(f"  MDL compressed:     {best_mdl_bits:.0f} bits")
        print(f"  Compression ratio:  {compression_ratio:.2f}x")
        
        if compression_ratio > 1:
            print(f"  ✅ Model found compressible structure!")
        else:
            print(f"  ⚠️  Data might be mostly noise")
    
    # =========================================================================
    # Model details
    # =========================================================================
    
    if verbose:
        # Fit the winning model
        winner_coeffs = np.polyfit(x, y, mdl_best)
        
        print()
        print("📐 WINNING MODEL DETAILS")
        print("-" * 40)
        print(f"  Selected degree: {mdl_best}")
        print(f"  Fitted coefficients: {np.round(winner_coeffs, 3)}")
        
        # Prediction error
        predictions = np.poly1d(winner_coeffs)(x)
        mse = np.mean((y - predictions)**2)
        rmse = np.sqrt(mse)
        
        print(f"  Training RMSE: {rmse:.4f}")
        print(f"  True noise std: {noise_std:.4f}")
        
        # Compare to true
        if mdl_best == true_degree:
            print()
            print("  🎯 MDL correctly identified the true structure!")
        elif mdl_best < true_degree:
            print()
            print("  ⚠️ MDL underfit - might need more data or less noise")
        else:
            print()
            print("  ⚠️ MDL overfit - true pattern might be simpler")
    
    if verbose:
        print()
        print("=" * 70)
        print("💡 KEY INSIGHT: MDL finds the model that COMPRESSES data best.")
        print("   Compression = Understanding. Random noise can't be compressed.")
        print("=" * 70)
    
    return {
        'true_degree': true_degree,
        'mdl_best': mdl_best,
        'aic_best': aic_best,
        'bic_best': bic_best,
        'mdl_correct': mdl_best == true_degree,
        'aic_correct': aic_best == true_degree,
        'bic_correct': bic_best == true_degree,
        'compression_ratio': compression_ratio,
        'mdl_scores': mdl_scores,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores
    }


def run_monte_carlo(
    true_degree: int = 2,
    n_trials: int = 100,
    noise_std: float = 1.5,
    n_samples: int = 50,
    max_degree: int = 10
):
    """
    Run multiple trials to compare MDL, AIC, BIC reliability.
    """
    print("\n" + "=" * 70)
    print("🎲 MONTE CARLO COMPARISON: MDL vs AIC vs BIC")
    print("=" * 70)
    print(f"\nRunning {n_trials} trials...")
    print(f"True degree: {true_degree}, Noise: {noise_std}, Samples: {n_samples}")
    
    mdl_correct = 0
    aic_correct = 0
    bic_correct = 0
    
    for i in range(n_trials):
        result = run_experiment(
            true_degree=true_degree,
            noise_std=noise_std,
            n_samples=n_samples,
            max_degree=max_degree,
            seed=i,
            verbose=False
        )
        
        mdl_correct += int(result['mdl_correct'])
        aic_correct += int(result['aic_correct'])
        bic_correct += int(result['bic_correct'])
    
    print()
    print("📊 ACCURACY OVER", n_trials, "TRIALS")
    print("-" * 40)
    print(f"  MDL: {mdl_correct}/{n_trials} ({100*mdl_correct/n_trials:.1f}%)")
    print(f"  AIC: {aic_correct}/{n_trials} ({100*aic_correct/n_trials:.1f}%)")
    print(f"  BIC: {bic_correct}/{n_trials} ({100*bic_correct/n_trials:.1f}%)")
    
    # Find winner
    winner = 'MDL' if mdl_correct >= max(aic_correct, bic_correct) else \
             ('AIC' if aic_correct > bic_correct else 'BIC')
    
    print()
    print(f"🏆 Winner: {winner}")
    print()


def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="MDL Principle - Model Selection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_minimal.py                     # Default settings
  python train_minimal.py --true-degree 3     # Use cubic polynomial
  python train_minimal.py --noise 0.5         # Low noise (easy)
  python train_minimal.py --noise 3.0         # High noise (hard)
  python train_minimal.py --monte-carlo       # Run 100 trials
        """
    )
    
    parser.add_argument('--true-degree', type=int, default=2,
                        help='True polynomial degree (default: 2)')
    parser.add_argument('--noise', type=float, default=1.5,
                        help='Noise standard deviation (default: 1.5)')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of data points (default: 50)')
    parser.add_argument('--max-degree', type=int, default=10,
                        help='Maximum degree to consider (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Run Monte Carlo comparison (100 trials)')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of Monte Carlo trials (default: 100)')
    
    args = parser.parse_args()
    
    if args.monte_carlo:
        run_monte_carlo(
            true_degree=args.true_degree,
            n_trials=args.trials,
            noise_std=args.noise,
            n_samples=args.samples,
            max_degree=args.max_degree
        )
    else:
        run_experiment(
            true_degree=args.true_degree,
            noise_std=args.noise,
            n_samples=args.samples,
            max_degree=args.max_degree,
            seed=args.seed,
            verbose=True
        )


if __name__ == "__main__":
    main()
