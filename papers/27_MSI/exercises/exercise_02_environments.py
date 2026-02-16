"""
Day 27: Machine Super Intelligence | Exercise 2
==============================================

Goal: Designing a Programmable Environment (μ) and Estimating K(μ).

Theoretical Context:
--------------------
In Exercise 1, we learned how to sum Upsilon. Now, we must understand the 
building blocks of that sum: the Environments.

In Shane Legg's Universal Intelligence framework, an environment is not 
just a "level" in a game. It is a **Computable Function**. Specifically:
- Input: Action sequence (a1, a2, ..., at)
- Output: Observation/Reward sequence (o1, r1, ..., ot, rt)

The fundamental law of the "Universal IQ Test" is that the weight of an 
environment is determined by its **Kolmogorov Complexity (K)**. 
K(μ) is the length of the shortest computer program that can implement μ.

The "Occam's Challenge":
-----------------------
If you are designing a test, you must know how 'hard' the questions are. 
In MSI, 'Hardness' == 'Complexity'. 
- A simple question: "What is 1+1?" (Very short program, low K).
- A complex question: "Simulate the fluid dynamics of a hurricane." (Very long program, high K).

Your Task:
----------
1. Complete the `BinarySearchEnv` class. 
2. Assign an appropriate K-score based on the 'Description Length' of the task.
3. Implement the `step()` method to provide the correct (A, O, R) feedback.

Pedagogical Insight:
--------------------
Recall the "Conveyor Belt" of Day 2. In this exercise, the environment acts 
as the "Signal Generator". The agent is trying to guess the hidden 
parameters of your code (the target number). If the search space is large, 
the "Source Code" for the environment is longer (requires more bits 
to specify the target), and thus the complexity K is higher.
"""

import sys
import os
from typing import Tuple, Any

# Ensure we can import the core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import Environment

# =============================================================================
# MASTERCLASS COMMENTARY: The Complexity of Search
# =============================================================================
# Why does range_size affect K?
# To specify a single number in a range of 1,000, you need roughly 10 bits 
# of information (log2(1000) ≈ 10).
# To specify a number in a range of 10, you only need ~3 bits.
#
# Therefore, an environment that hides a number in a large range is 
# physically a more "complex" program because its initial state (the target) 
# takes more code/memory to define.
# =============================================================================

class BinarySearchEnv(Environment):
    """
    A 'Number Guessing' environment.
    
    The agent submits a number (Action).
    The environment returns if the target is higher, lower, or found.
    """
    def __init__(self, range_size: int = 16, target: int = 7):
        """
        Args:
            range_size: The total count of possible numbers (0 to range_size-1).
            target: The hidden number the agent is looking for.
        """
        # TODO: Assign complexity K. 
        # Logic: A good proxy for K in search tasks is ceil(log2(range_size)).
        # It represents the 'information content' of the hidden state.
        import math
        k_score = int(math.ceil(math.log2(range_size)))
        
        super().__init__(complexity=k_score) 
        self.target = target
        self.range_size = range_size
        self.tries = 0

    def reset(self) -> Any:
        """
        Resets the game state. 
        Returns the initial 'empty' observation.
        """
        self.tries = 0
        return "Start guessing!"

    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Processes the guess and returns feedback.
        
        Instructional Logic:
        --------------------
        1. Increment the `tries` counter.
        2. If action < target: Observation = "Higher" (Agent should guess higher next).
        3. If action > target: Observation = "Lower".
        4. If action == target: Observation = "Correct", Reward = 1.0, Done = True.
        
        Args:
            action: The agent's current guess.
            
        Returns:
            (Observation, Reward, Done)
        """
        self.tries += 1
        
        # [YOUR CODE HERE]
        # Implement the comparison logic...
        
        # Default placeholder return:
        obs, reward, done = "Not implemented", 0.0, False
        
        return obs, reward, done

# =============================================================================
# VERIFICATION SUITE: Testing the Induction Loop
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 27 EXERCISE 2: DESIGNING ENVIRONMENTS")
    print("=" * 60)
    
    # We create a 16-slot search world. K should be 4.
    print("\n[STEP 1] Initializing BinarySearchEnv(size=16, target=3)...")
    env = BinarySearchEnv(range_size=16, target=3)
    
    print(f"  Complexity Assigned (K): {env.complexity}")
    assert env.complexity == 4, f"Complexity should be 4 for size 16. Got {env.complexity}"
    print("  [PASS] Complexity properly mapped to information content.")

    print("\n[STEP 2] Simulating Agent Interaction...")
    obs = env.reset()
    
    # Guess 1: The middle of the range
    print("  Action: Guess 1")
    obs, rew, done = env.step(1)
    print(f"  Observation: {obs}")
    assert obs == "Higher", "Target is 3, guess was 1. Observation must be 'Higher'."
    
    # Guess 2: Correct
    print("  Action: Guess 3")
    obs, rew, done = env.step(3)
    print(f"  Observation: {obs} | Reward: {rew}")
    
    assert rew == 1.0, "Correct guess should yield 1.0 reward."
    assert done == True, "Episode should end after correct guess."
    print("  [PASS] Feedback logic (A, O, r) is consistent.")

    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE: You are now an Environment Architect.")
    print("=" * 60)
