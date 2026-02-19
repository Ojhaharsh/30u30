"""
solution_04_synthetic_oracle.py - Day 30: RLHF
Goal: Implement the Synthetic Oracle (simulated teacher).
"""

import numpy as np

class SyntheticOracle:
    def __init__(self):
        """
        Simulates a human evaluator using ground truth rewards.
        """
        pass
        
    def query(self, segment1_rewards, segment2_rewards):
        """
        Returns 0 if segment1 is better, 1 if segment2 is better.
        Uses sum of rewards as the ground truth preference.
        """
        sum1 = np.sum(segment1_rewards)
        sum2 = np.sum(segment2_rewards)
        if sum1 > sum2:
            return 0
        else:
            return 1
            
if __name__ == "__main__":
    oracle = SyntheticOracle()
    s1 = [1, 1, 1] 
    s2 = [0, 0, 0] 
    
    choice = oracle.query(s1, s2)
    print(f"Choice: {choice}")
