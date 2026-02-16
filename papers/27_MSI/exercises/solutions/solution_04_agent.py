import sys
import os
from typing import Any, List

# Ensure we can import the core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import Agent

class PredictiveHistoryAgent(Agent):
    """
    Solution for the 'Compression' Agent.
    """
    def __init__(self, history_len: int = 10):
        self.history = []
        self.transitions = {} 
        self.last_observation = None

    def act(self, observation: Any) -> int:
        # Prediction Logic: If we know the pattern, skip the trial-and-error
        if observation in self.transitions:
            return self.transitions[observation]
        
        # Default fallback
        return 0

    def observe(self, observation: Any, reward: float, done: bool):
        # Induction Logic: Linking the sequence of truth
        if self.last_observation is not None:
            self.transitions[self.last_observation] = observation
        
        self.last_observation = observation
        self.history.append(observation)

if __name__ == "__main__":
    agent = PredictiveHistoryAgent()
    # Mocking learning a [0, 1, 2] pattern
    agent.observe(0, 0, False)
    agent.observe(1, 1, False)
    # At state 0, it should now predict 1
    assert agent.act(0) == 1
    print("[OK] Solution 4 verified.")
