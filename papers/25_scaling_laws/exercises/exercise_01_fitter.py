"""
Day 25 Exercise 1: Power-Law Fitter

Your task is to implement the logic for fitting a power law L(N) = A * N^-alpha.

This is the mathematical foundation of Scaling Laws. By the end of this exercise,
you will understand how to transform raw loss data into predictable laws.

Instructions:
1. Implement the log-transformation of the data.
2. Use np.polyfit to perform linear regression.
3. Extract alpha and the intercept.

Success Criteria:
- alpha should be approximately 0.076 for the provided data.
"""

import numpy as np
import matplotlib.pyplot as plt

def fit_power_law(ns, ls):
    """
    Fits L = A * N^-alpha
    
    TODO: 
    1. Transform ns and ls to log10 space.
    2. Use np.polyfit(log_ns, log_ls, 1) to find the slope and intercept.
    3. Return alpha (-slope) and the constant A (10^intercept).
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    # Baseline data points (N, L)
    ns = [1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]
    ls = [3.20, 2.85, 2.70, 2.40, 2.28, 2.05, 1.95]
    
    print("Fitting Scaling Law to empirical data...")
    alpha, A = fit_power_law(ns, ls)
    
    if alpha is not None:
        print(f"[OK] Fitted alpha: {alpha:.4f}")
        print(f"[OK] Fitted Constant A: {A:.4f}")
        print(f"Resulting Law: L(N) = {A:.2f} * N^-{alpha:.4f}")
    else:
        print("[FAIL] fit_power_law not implemented.")
