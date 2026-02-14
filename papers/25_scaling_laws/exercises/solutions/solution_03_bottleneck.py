"""
Day 25 Solution 3: Overfitting Detector

This solution implements the heuristic that D must scale as O(N^0.74) to avoid
early overfitting for the WebText dataset used in the paper.
"""

import math

def calculate_d_opt(n_params):
    """
    D_opt approx 5000 * n_params^0.74 
    Reference: Section 4, Figure 5 Analysis.
    """
    return 5000 * math.pow(n_params, 0.74)

def check_bottleneck(n_params, d_size):
    """
    Determines if the dataset size is sufficient for the model size.
    """
    d_opt = calculate_d_opt(n_params)
    
    if d_size < d_opt:
        return f"[FAIL] D-BOTTLENECKED (Needs {d_opt/1e6:.1f}M tokens)"
    else:
        return f"[OK] N-SCALING (Headroom: { (d_size - d_opt)/1e6 :.1f}M tokens)"

if __name__ == "__main__":
    # Test cases
    cases = [
        (1e6, 1e7),   # 10M tokens for 1M params -> Likely bottlenecked
        (1e6, 1e9),   # 1B tokens for 1M params -> Deeply in scaling regime
        (175e9, 300e9) # GPT-3 -> Significant D-bottleneck by Kaplan standards
    ]
    
    print(f"{'N':<12} | {'D':<12} | {'Status':<20}")
    print("-" * 60)
    for n, d in cases:
        status = check_bottleneck(n, d)
        print(f"{n:<12.1e} | {d:<12.1e} | {status}")
