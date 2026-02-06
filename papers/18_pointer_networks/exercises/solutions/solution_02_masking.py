"""
Solution 2: Implement Attention Masking
Standardized for Day 18 of 30u30

Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch
import torch.nn.functional as F

def apply_mask(scores, mask):
    """
    Apply boolean mask to attention scores and return log_probs.
    """
    # Replace masked positions with -inf so they become 0 in softmax
    masked_scores = scores.masked_fill(mask, float('-inf'))
    return F.log_softmax(masked_scores, dim=-1)
