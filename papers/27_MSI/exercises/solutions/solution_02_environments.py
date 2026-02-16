import sys
import os
import math
from typing import Tuple, Any

# Ensure we can import the core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import Environment

class BinarySearchEnv(Environment):
    """
    Solution for the 'Number Guessing' environment.
    """
    def __init__(self, range_size: int = 16, target: int = 7):
        # K is the 'Information Content' of the hidden target.
        # We use ceil(log2(range_size)) as a proxy for description complexity.
        k_score = int(math.ceil(math.log2(range_size)))
        super().__init__(complexity=k_score) 
        self.target = target
        self.range_size = range_size
        self.tries = 0

    def reset(self) -> Any:
        self.tries = 0
        return "Start guessing!"

    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Implementation of the (A, O, R) search logic.
        """
        self.tries += 1
        
        if action < self.target:
            return "Higher", 0.0, False
        elif action > self.target:
            return "Lower", 0.0, False
        else:
            # Found the target!
            return "Correct", 1.0, True

if __name__ == "__main__":
    env = BinarySearchEnv(range_size=16, target=3)
    obs, rew, done = env.step(1)
    assert obs == "Higher"
    obs, rew, done = env.step(3)
    assert rew == 1.0
    print("[OK] Solution 2 verified.")
