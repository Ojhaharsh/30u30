import numpy as np
from typing import List

def calculate_upsilon(rewards: List[float], complexities: List[int]) -> float:
    """
    Implements the Universal Intelligence weighted summation.
    
    Formula: Σ (Reward_μ * 2^-K_μ)
    """
    total_score = 0.0
    
    # Pythonic zip allows us to iterate through the Agent-Env alignment
    for reward, k in zip(rewards, complexities):
        # The Universal Prior: 2^-K
        # This weight decreases exponentially as complexity increases.
        weight = 2.0**(-k)
        
        # Adding the weighted performance contribution
        total_score += weight * reward
        
    return total_score

if __name__ == "__main__":
    # Test mirroring Exercise 1 scenario
    rewards = [0.9, 0.9]
    complexities = [2, 12]
    score = calculate_upsilon(rewards, complexities)
    
    expected = (2.0**-2 * 0.9) + (2.0**-12 * 0.9)
    print(f"Calculated Score: {score:.8f}")
    assert np.isclose(score, expected)
    print("[OK] Solution 1 verified.")
