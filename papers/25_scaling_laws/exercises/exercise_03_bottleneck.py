"""
Day 25 Exercise 3: Overfitting Detector

Kaplan et al. show that if the dataset size D is too small relative to N,
the model will start to overfit, and performance will stop following the power law.

Your task is to implement a detector that finds the "Optimal D" for a given N
to stay in the scaling regime.

Reference: Section 4, "Charting the Infinite Data Limit".

Equation 4.3 (empirical estimate): 
D_opt approx 5e3 * N^0.74 (for this specific dataset)

Instructions:
1. Implement the calculate_d_opt function.
2. Given a list of (N, D) pairs, flag which ones are at risk of overfitting.
"""

def calculate_d_opt(n_params):
    """
    TODO: Implement the approximation for D_opt.
    D_opt approx 5000 * n_params^0.74
    """
    # YOUR CODE HERE
    pass

def check_bottleneck(n_params, d_size):
    """
    TODO: Calculate d_opt and compare with d_size.
    If d_size < d_opt, return "[FAIL] D-BOTTLENECKED"
    Else, return "[OK] N-SCALING"
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    # Test cases
    cases = [
        (1e6, 1e7),   # 1M params, 10M tokens
        (1e6, 1e9),   # 1M params, 1B tokens
        (175e9, 300e9) # GPT-3 size, Standard dataset
    ]
    
    print(f"{'N':<12} | {'D':<12} | {'Status':<20}")
    print("-" * 50)
    for n, d in cases:
        status = check_bottleneck(n, d)
        if status:
            print(f"{n:<12.1e} | {d:<12.1e} | {status}")
        else:
            print(f"{n:<12.1e} | {d:<12.1e} | [FAIL] Not implemented")
