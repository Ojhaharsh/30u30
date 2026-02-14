"""
Day 25 Exercise 2: The 6N Estimator

Your task is to implement the training compute estimation formula:
C approx 6 * N * total_tokens

This formula is the industry standard for estimating how many GPUs and how much
time you need to train a model.

Reference: Equation 2.1 in Kaplan et al. (2020).

Instructions:
1. Implement the estimate_compute function.
2. Calculate the PF-days (PetaFLOP-days) for the given models.
"""

def estimate_compute(n_params, tokens):
    """
    TODO: Implement the 6N rule.
    Return the total floating point operations (FLOPs).
    """
    # YOUR CODE HERE
    pass

def to_pf_days(flops):
    """
    Converts raw FLOPs to PetaFLOP-days.
    1 PF-day = 10^15 * 60 * 60 * 24 FLOPs.
    """
    return flops / (1e15 * 60 * 60 * 24)

if __name__ == "__main__":
    # Case 1: GPT-2 Small (117M params, trained on 40B tokens)
    n_gpt2 = 117e6
    tokens_gpt2 = 40e9
    
    c_gpt2 = estimate_compute(n_gpt2, tokens_gpt2)
    
    if c_gpt2 is not None:
        print(f"GPT-2 Estimate: {c_gpt2:.2e} FLOPs")
        print(f"GPT-2 PF-days: {to_pf_days(c_gpt2):.2f}")
    else:
        print("[FAIL] estimate_compute not implemented.")
