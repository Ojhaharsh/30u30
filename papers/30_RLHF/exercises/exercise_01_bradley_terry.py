"""
exercise_01_bradley_terry.py - Day 30: RLHF
Goal: Implement the Bradley-Terry model for preference probability.

Reference: Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"
Eq 1: P(1 > 2) = exp(r1) / (exp(r1) + exp(r2))
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_preference_probability(sum_r1, sum_r2):
    """
    Computes the probability that segment 1 is preferred over segment 2
    using the Bradley-Terry model.
    
    P(1 > 2) = exp(sum_r1) / (exp(sum_r1) + exp(sum_r2))
    
    Args:
        sum_r1: Scalar tensor/float, sum of rewards for segment 1
        sum_r2: Scalar tensor/float, sum of rewards for segment 2
        
    Returns:
        prob: Float, probability that segment 1 is preferred
    """
    # TODO: Implement the Bradley-Terry formula
    pass

if __name__ == "__main__":
    r1 = torch.tensor(1.0)
    r2 = torch.tensor(0.0)
    
    prob = compute_preference_probability(r1, r2)
    print(f"P(1 > 2) with r1=1, r2=0: {prob}")
    
    # Expected: exp(1)/(exp(1)+exp(0)) = 2.718 / 3.718 = 0.731
    expected = np.exp(1) / (np.exp(1) + 1)
    
    if prob is not None and abs(prob - expected) < 1e-3:
        print("Test Passed!")
    else:
        print("Test Failed. Keep trying!")
