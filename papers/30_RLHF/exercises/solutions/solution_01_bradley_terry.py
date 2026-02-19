"""
solution_01_bradley_terry.py - Day 30: RLHF
Goal: Implement the Bradley-Terry model for preference probability.
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_preference_probability(sum_r1, sum_r2):
    """
    Computes the probability that segment 1 is preferred over segment 2
    using the Bradley-Terry model.
    """
    # Eq 1 from Christiano et al. (2017)
    # P(1 > 2) = exp(r1) / (exp(r1) + exp(r2))
    
    # Use sum_r1 and sum_r2 directly as logits
    exp1 = torch.exp(sum_r1)
    exp2 = torch.exp(sum_r2)
    prob = exp1 / (exp1 + exp2)
    return prob

if __name__ == "__main__":
    r1 = torch.tensor(1.0)
    r2 = torch.tensor(0.0)
    
    prob = compute_preference_probability(r1, r2)
    print(f"P(1 > 2) with r1=1, r2=0: {prob}")
