"""
Day 25 Solution 1: Power-Law Fitter

This solution demonstrates how to use the relationship log(L) = log(A) - alpha * log(N)
 to extract the scaling exponents discovered by Kaplan et al. (2020).
"""

import numpy as np
import matplotlib.pyplot as plt

def fit_power_law(ns, ls):
    """
    Fits L = A * N^-alpha by performing linear regression in log-log space.
    Reference: Kaplan et al. (2020), Section 2.2.
    """
    # 1. Transform to log space
    log_ns = np.log10(ns)
    log_ls = np.log10(ls)
    
    # 2. Linear regression (degree 1 polyfit)
    # y = mx + c  =>  log(L) = -alpha * log(N) + log(A)
    coeffs = np.polyfit(log_ns, log_ls, 1)
    
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # 3. Extract parameters
    alpha = -slope
    A = 10**intercept
    
    return alpha, A

if __name__ == "__main__":
    # Baseline data points (N, L)
    ns = np.array([1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8])
    ls = np.array([3.20, 2.85, 2.70, 2.40, 2.28, 2.05, 1.95])
    
    print("Fitting Scaling Law to empirical data...")
    alpha, A = fit_power_law(ns, ls)
    
    print(f"[OK] Fitted alpha: {alpha:.4f}")
    print(f"[OK] Fitted Constant A: {A:.4f}")
    print(f"Resulting Law: L(N) = {A:.2f} * N^-{alpha:.4f}")
    
    # Verification against Kaplan's alpha_N approx 0.076
    diff = abs(alpha - 0.076) / 0.076
    if diff < 0.15:
        print(f"[OK] Exponent is consistent with Kaplan et al. (2020) Table 1.")
    else:
        print(f"[NOTE] Exponent {alpha:.4f} deviates from paper's 0.076. This is expected for small-scale synthetic data.")
