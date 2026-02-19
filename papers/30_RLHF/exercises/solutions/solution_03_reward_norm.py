"""
solution_03_reward_norm.py - Day 30: RLHF
Goal: Implement running mean/std normalization for learned rewards.
"""

import numpy as np

class RewardNormalizer:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        
    def update(self, rewards_batch):
        """
        Updates running statistics using Welford's online algorithm or simple batch update.
        Here we use simple batch update for clarity, though Welford is better for streams.
        """
        batch_mean = np.mean(rewards_batch)
        batch_var = np.var(rewards_batch)
        batch_count = len(rewards_batch)
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count
        
        # Update var (simplified, assumes large batches)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, rewards):
        return (rewards - self.mean) / np.sqrt(self.var + 1e-8)
        
if __name__ == "__main__":
    rn = RewardNormalizer()
    rn.update(np.array([1.0, 2.0, 3.0]))
    print(f"Mean: {rn.mean}, Var: {rn.var}")
    print(f"Norm(2.0): {rn.normalize(np.array([2.0]))}")
