"""
Day 27: Machine Super Intelligence | Exercise 5
==============================================

Goal: Computational Proof of the Invariance Theorem.

Theoretical Context:
--------------------
You have implemented the Sum, the Env, the Strategy, and the Induction. 
Now, we address the final objection to Universal Intelligence: 
> "What if I use C++ instead of Python? Won't my complexities (K) change?"

The answer is the **Invariance Theorem**.

In Solomonoff's Theory of Induction (and Legg's thesis), the absolute 
complexity K(μ) depends on the choice of the Universal Turing Machine (UTM) 
used for measurement. 

However, the theorem states:
    K_UTM1(μ) = K_UTM2(μ) + C

Where C is a constant (the length of the 'Translation Program' between 
the two languages). This means that while absolute scores change, 
the **relative rankings** of intelligence remain stable across ANY 
sufficiently powerful language.

The "Universal Metric" Stability:
---------------------------------
If Agent A is smarter than Agent B in Python, Agent A is also smarter 
in Brainfuck (modulo the overhead of translating Brainfuck).

Your Task:
----------
1. Calculate a base Upsilon score for an agent.
2. Simulate a language shift (UTM change) by adding a constant C to all K-scores.
3. Verify that the new score is exactly scaled by 2^-C.
4. Prove that the *ratio* of intelligence between two agents is invariant 
   to the language shift.

Pedagogical Insight:
--------------------
Recall the "Vanishing Gradient" problem in Day 2. LSTMs solve it by 
ensuring information flows via addition. Similarly, the Invariance Theorem 
ensures that the "Signal of Intelligence" flows through any UTM without 
being lost to narrow linguistic choices.
"""

import numpy as np

# =============================================================================
# MASTERCLASS COMMENTARY: The Constant of Nature
# =============================================================================
# Legg notes that for "Narrow" UTMs (like a calculator), C can be huge. 
# But for "General" UTMs (like Python, C++, or Human Logic), C is small 
# compared to the complexity of the environments we care about. 
#
# This exercise proves that Upsilon is 'asymptotically objective'. 
# It doesn't matter what your native tongue is; logic is universal.
# =============================================================================

def calculate_upsilon(rewards, complexities):
    """
    Standard weighted summation logic.
    Σ (Reward * 2^-K)
    """
    return sum(r * (2.0**-k) for r, k in zip(rewards, complexities))

def analyze_invariance():
    """
    Verifies that shifting complexities by a constant C scales the final 
    score by exactly 2^-C.
    
    Instructional Logic:
    --------------------
    1. Define a base set of environments and rewards.
    2. Calculate Upsilon_Base.
    3. Add a constant C=3 to every task in complexities.
    4. Calculate Upsilon_Shifted.
    5. Show that Upsilon_Shifted / Upsilon_Base == 0.125 (which is 2^-3).
    """
    
    # Task Complexities in 'Universal Machine 1' (e.g. Python)
    k_utm1 = [2, 5, 8, 12]
    # Agent performance
    rewards = [0.8, 0.6, 0.4, 0.2]
    
    print("\n[UTM 1] Language: Python")
    upsilon_1 = calculate_upsilon(rewards, k_utm1)
    print(f"  Upsilon_1: {upsilon_1:.6f}")

    # Simulate shifting to 'Universal Machine 2' (e.g. Brainfuck)
    # Let's say Brainfuck takes 3 bits longer to implement any concept.
    C = 3
    k_utm2 = [k + C for k in k_utm1]
    
    print(f"\n[UTM 2] Language: Translated (Shift C={C})")
    upsilon_2 = calculate_upsilon(rewards, k_utm2)
    print(f"  Upsilon_2: {upsilon_2:.6f}")

    # The Mathematical Verification:
    # Logic: 2^-(K+C) = 2^-K * 2^-C
    # Therefore: Σ (2^-(K+C) * V) = 2^-C * Σ (2^-K * V)
    
    ratio = upsilon_2 / upsilon_1
    expected_ratio = 2.0**-C
    
    print(f"\n  Found Ratio: {ratio:.4f}")
    print(f"  Expected Ratio (2^-{C}): {expected_ratio:.4f}")
    
    if np.isclose(ratio, expected_ratio):
        print("  [PASS] Scaling Invariance Confirmed.")
    else:
        print("  [FAIL] Invariance logic broken.")
        return False
        
    print("\n[STEP 3] Verifying Ranking Stability...")
    # Second Agent
    rewards_b = [0.2, 0.4, 0.6, 0.8]
    u1_b = calculate_upsilon(rewards_b, k_utm1)
    u2_b = calculate_upsilon(rewards_b, k_utm2)
    
    # Ratios should be identical
    if (upsilon_1 > u1_b) == (upsilon_2 > u2_b):
        print("  [PASS] Agent ranking is stable across languages.")
    else:
        print("  [FAIL] Choice of language changed the 'smarter' agent!")
        return False
        
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 27 EXERCISE 5: THE INVARIANCE THEOREM")
    print("=" * 60)
    
    if analyze_invariance():
        print("\n" + "=" * 60)
        print("EXERCISE COMPLETE: You have proven that Intelligence is Universal.")
        print("=" * 60)
    else:
        exit(1)
