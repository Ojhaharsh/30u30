"""
exercise_02_preference_loss.py - Day 30: RLHF
Goal: Implement the Preference Cross-Entropy Loss (Eq 2).

Reference: Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"
Eq 2: Loss = - (1-P)log(1-P_pred) - P log(P_pred)
"""

import torch
import torch.nn.functional as F

def preference_loss(reward_sum1, reward_sum2, label):
    """
    Computes the cross-entropy loss for a batch of preferences.
    label[i] = 0 means segment 1 is preferred.
    label[i] = 1 means segment 2 is preferred.
    
    Args:
        reward_sum1: Tensor (batch,)
        reward_sum2: Tensor (batch,)
        label: Tensor (batch,) (long)
        
    Returns:
        loss: Scalar tensor
    """
    # TODO: Stack rewards, compute logits, use cross_entropy
    pass
    
if __name__ == "__main__":
    r1 = torch.tensor([1.0, 0.0])
    r2 = torch.tensor([0.0, 1.0])
    labels = torch.tensor([0, 1]) # First pair: 1>2, Second pair: 2>1 (both "correct" given rewards)
    
    loss = preference_loss(r1, r2, labels)
    print(f"Loss: {loss}")
    
    # Validation
    if loss is not None and loss < 0.4: # Should be low
        print("Test Passed!")
    else:
        print("Test Failed/Not Implemented")
