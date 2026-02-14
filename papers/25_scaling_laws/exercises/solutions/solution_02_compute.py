"""
Day 25 Solution 2: The 6N Estimator

This solution implements the 6N rule: 
Training compute C is approximately 6 * N * total_tokens.
"""

def estimate_compute(n_params, tokens):
    """
    Reference: Kaplan et al. (2020), Equation 2.1.
    C approx 6 * N * tokens.
    """
    return 6.0 * n_params * tokens

def to_pf_days(flops):
    """
    Converts raw FLOPs to PetaFLOP-days.
    1 PF-day = 10^15 * (60 * 60 * 24) FLOPs.
    """
    return flops / (1e15 * 60 * 60 * 24)

if __name__ == "__main__":
    # Case 1: GPT-2 Small (117M params, trained on 40B tokens)
    n_gpt2 = 117e6
    tokens_gpt2 = 40e9
    
    c_gpt2 = estimate_compute(n_gpt2, tokens_gpt2)
    print(f"GPT-2 Estimate: {c_gpt2:.2e} FLOPs")
    print(f"GPT-2 PF-days: {to_pf_days(c_gpt2):.2f}")
    
    # Case 2: Comparison with GPT-3 (175B params, trained on 300B tokens)
    n_gpt3 = 175e9
    tokens_gpt3 = 300e9
    c_gpt3 = estimate_compute(n_gpt3, tokens_gpt3)
    print(f"\nGPT-3 Estimate: {c_gpt3:.2e} FLOPs")
    print(f"GPT-3 PF-days: {to_pf_days(c_gpt3):.2f}")
