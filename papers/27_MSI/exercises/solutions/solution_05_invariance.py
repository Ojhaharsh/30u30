import numpy as np

def calculate_upsilon(rewards, complexities):
    """Σ (Reward * 2^-K)"""
    return sum(r * (2.0**-k) for r, k in zip(rewards, complexities))

def analyze_invariance():
    """
    Verification of the scaling property: U(K+C) = U(K) * 2^-C
    """
    k_utm1 = [2, 5, 8, 12]
    rewards = [0.8, 0.6, 0.4, 0.2]
    
    ups1 = calculate_upsilon(rewards, k_utm1)
    
    C = 3
    k_utm2 = [k + C for k in k_utm1]
    ups2 = calculate_upsilon(rewards, k_utm2)
    
    ratio = ups2 / ups1
    expected = 2.0**-C
    
    # Mathematical Proof:
    # 2^-(K+C) = 2^-K * 2^-C
    # U2 = Σ V * 2^-(K+C) = 2^-C * Σ V * 2^-K = 2^-C * U1
    
    assert np.isclose(ratio, expected)
    return True

if __name__ == "__main__":
    if analyze_invariance():
        print("[OK] Solution 5 verified.")
