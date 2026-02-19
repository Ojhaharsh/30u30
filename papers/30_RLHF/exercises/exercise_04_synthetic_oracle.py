"""
exercise_04_synthetic_oracle.py - Day 30: RLHF
Goal: Implement the Synthetic Oracle (simulated teacher).
The Oracle prefers segments with higher ground-truth environment rewards.
"""

class SyntheticOracle:
    def __init__(self):
        """
        Simulates a human evaluator using ground truth rewards.
        """
        pass
        
    def query(self, segment1_rewards, segment2_rewards):
        """
        Returns 0 if segment1 is better, 1 if segment2 is better.
        
        Args:
            segment1_rewards: List of scalar rewards for segment 1.
            segment2_rewards: List of scalar rewards for segment 2.
            
        Returns:
            start_index: int (0 or 1)
        """
        # TODO: Implement the preference logic (sum of rewards)
        pass
        
if __name__ == "__main__":
    oracle = SyntheticOracle()
    s1 = [1, 1, 1] # sum 3
    s2 = [0, 0, 0] # sum 0
    
    choice = oracle.query(s1, s2)
    print(f"Choice: {choice}")
    
    if choice == 0:
        print("Test Passed!")
    else:
        print("Test Failed/Not Implemented")
