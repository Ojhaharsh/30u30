"""
Day 27: Machine Super Intelligence | Exercise 1
==============================================

Goal: Implementation of the Universal Intelligence (Υ) Summation.

Theoretical Context:
--------------------
In his 2008 thesis, Shane Legg argues that intelligence is not a monolithic 
trait, but an emergence from a generalist's ability to navigate a 
mathematical space of tasks. 

To measure this without human bias, he uses the "Universal Prior" from 
Algorithmic Information Theory (Day 26). This prior weights every 
computable environment μ by 2^-K(μ).

The formula you are about to implement is:
    Υ(π) = Σ_{μ ∈ E} 2^-K(μ) * V_μ^π

Where:
- π is the agent (the policy we are testing).
- μ is a specific environment (a computer program).
- K(μ) is the Kolmogorov complexity (how many lines of code describe μ).
- V_μ^π is the expected reward the agent gets in that environment.

Pedagogical Insight:
--------------------
Think of this as an "Infinite Decathlon." In a normal decathlon, we 
average your score across 10 events. In Upsilon, we average your score 
across EVERY POSSIBLE PROGRAM. But because there are infinite programs, 
we need the 2^-K weighting to ensure the sum stays finite (converges) 
and that fundamental patterns (simple code) matter more than noise.

Your Task:
----------
1. Implement the `calculate_upsilon` function.
2. Handle the exponential weighting factor carefully.
3. Understand the difference between 'Narrow' and 'Universal' intelligence.

Reference: Shane Legg, "Machine Super Intelligence", Chapter 4.
"""

import numpy as np
from typing import List

# =============================================================================
# MASTERCLASS COMMENTARY: The Weighted Sum
# =============================================================================
# In Day 2 (LSTMs), we saw how the "cell state" acts as a conveyor belt 
# for information. Here, the "Universal Prior" (2^-K) acts as a 
# conveyor belt for "Truth". 
#
# If K is small, the environment is 101010... (a simple pattern).
# If K is large, the environment is a chaotic weather system.
#
# Legg's insight: If you can't solve the simple patterns, you aren't 
# intelligent. Period.
# =============================================================================

def calculate_upsilon(rewards: List[float], complexities: List[int]) -> float:
    """
    Calculates the Universal Intelligence score.
    
    Instructional Step-by-Step:
    ---------------------------
    1. Initialize a `total_score` variable to 0.0.
    2. Loop through each reward (V_μ) and its corresponding complexity (K_μ).
    3. Calculate the weight W = 2 raised to the power of (-K_μ).
    4. Calculate the 'Weighted Contribution': Contribution = W * V_μ.
    5. Accumulate this into the total.
    
    Args:
        rewards: A list of expected rewards [0, 1] for different environments.
        complexities: The K-score (integer) for each environment.
        
    Returns:
        The final Υ(π) scalar.
    """
    total_score = 0.0
    
    # [YOUR CODE HERE]
    # Implementation Hint: Use np.power(2.0, -K) or 2.0**(-K).
    # ensure your rewards are treated as floats to prevent precision loss.
    
    return total_score


# =============================================================================
# VERIFICATION SUITE: Generalist vs Specialist
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 27 EXERCISE 1: THE UNIVERSAL METRIC")
    print("=" * 60)
    
    print("\n[SCENARIO 1] The Bias of Simplicity")
    # We have two environments:
    # A: "Hello World" (Complexity K=2)
    # B: "Crytographic Hashing" (Complexity K=12)
    # Even if the rewards are the same, Environment A is 1024x more important 
    # for the intelligence score.
    
    test_rewards = [0.9, 0.9]
    test_complexities = [2, 12]
    
    score = calculate_upsilon(test_rewards, test_complexities)
    
    # Detailed derivation matching Day 2's walkthrough style:
    # Term 1 (K=2): 2^-2 * 0.9 = 0.25 * 0.9 = 0.225
    # Term 2 (K=12): 2^-12 * 0.9 = 0.000244 * 0.9 = 0.0002196
    # Total: 0.2252196
    expected = (2.0**-2 * 0.9) + (2.0**-12 * 0.9)
    
    print(f"  Calculated Upsilon: {score:.8f}")
    if np.isclose(score, expected, atol=1e-8):
        print("  [PASS] Mathematical precision verified.")
    else:
        print(f"  [FAIL] Expected {expected:.8f}, got {score:.8f}")

    print("\n[SCENARIO 2] The Generalist's Revenge")
    # Agent Alpha: Solves simple tasks perfect, fails hard ones.
    # Agent Beta: Fails simple tasks, solves hard ones perfect.
    # Alpha should have a MUCH higher Upsilon.
    
    alpha_upsilon = calculate_upsilon([1.0, 0.0], [3, 10]) # Perfect on K=3
    beta_upsilon = calculate_upsilon([0.0, 1.0], [3, 10])  # Perfect on K=10
    
    print(f"  Agent Alpha (Generalist): {alpha_upsilon:.6f}")
    print(f"  Agent Beta (Specialist):  {beta_upsilon:.6f}")
    
    if alpha_upsilon > beta_upsilon:
        print("  [PASS] Hierarchy of Generalization confirmed.")
    else:
        print("  [FAIL] The specialist should not outscore the generalist.")

    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE: You have implemented the 'Gold Standard' of AGI measurement.")
    print("=" * 60)
