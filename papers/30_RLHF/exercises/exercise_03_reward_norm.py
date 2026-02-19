"""
exercise_03_reward_norm.py - Day 30: RLHF
Goal: Implement running mean/std normalization for learned rewards.
Since we use preferences, the RM can drift; normalization provides a stable scale for PPO.
"""

import numpy as np

class RewardNormalizer:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        
    def update(self, rewards_batch):
        """
        Updates the running mean and variance using Welford's algorithm
        or a batch update.
        Args:
            rewards_batch: List or array of scalar rewards.
        """
        # TODO: Update self.mean and self.var using the new batch
        pass
        
    def normalize(self, rewards):
        """
        Returns (rewards - mean) / sqrt(var + 1e-8)
        """
        # TODO: Implement normalization
        pass
        
if __name__ == "__main__":
    rn = RewardNormalizer()
    batch1 = np.array([1.0, 2.0, 3.0])
    rn.update(batch1)
    
    norm = rn.normalize(np.array([2.0]))
    # Mean should be 2.0. Std should be sqrt(2/3) ~= 0.816 (population) or 1.0 (sample)
    # If using simple batch update, check logic
    
    print(f"Norm(2.0): {norm}")
    
    if norm is not None and abs(norm) < 0.1: # 2.0 is the mean
        print("Test Passed!")
    else:
        print("Test Failed/Not Implemented")
