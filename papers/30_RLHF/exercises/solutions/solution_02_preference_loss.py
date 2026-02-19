"""
solution_02_preference_loss.py - Day 30: RLHF
Goal: Implement the Preference Cross-Entropy Loss (Eq 2).
"""

import torch
import torch.nn.functional as F

def preference_loss(reward_sum1, reward_sum2, label):
    """
    Computes the cross-entropy loss for a batch of preferences.
    """
    # Stack rewards to form logits of shape (batch, 2)
    # logits[i, 0] = reward_sum1[i]
    # logits[i, 1] = reward_sum2[i]
    logits = torch.stack([reward_sum1, reward_sum2], dim=1)
    
    # Cross entropy loss between logits and integer labels
    loss = F.cross_entropy(logits, label)
    return loss
    
if __name__ == "__main__":
    r1 = torch.tensor([1.0, 0.0])
    r2 = torch.tensor([0.0, 1.0])
    labels = torch.tensor([0, 1])
    
    loss = preference_loss(r1, r2, labels)
    print(f"Loss: {loss}")
