import numpy as np
import sys
import os
from typing import Any

# Ensure we can import the core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import Agent

class EpsilonGreedyAgent(Agent):
    """
    Solution for the ε-greedy strategy.
    """
    def __init__(self, epsilon: float = 0.1, actions: int = 4):
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}

    def act(self, observation: Any) -> int:
        state = str(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions)
        
        # Branching between Exploration and Exploitation
        if np.random.random() < self.epsilon:
            # EXPLORE: Random action from the space of possibilities
            return np.random.randint(0, self.actions)
        else:
            # EXPLOIT: Best known action per current Q-values
            return np.argmax(self.q_table[state])

    def observe(self, observation: Any, reward: float, done: bool):
        pass

if __name__ == "__main__":
    agent = EpsilonGreedyAgent(epsilon=0.0)
    agent.q_table["s1"] = np.array([0, 100, 0, 0])
    assert agent.act("s1") == 1
    print("[OK] Solution 3 verified.")
