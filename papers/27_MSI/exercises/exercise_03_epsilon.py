"""
Day 27: Machine Super Intelligence | Exercise 3
==============================================

Goal: Investigating the Exploration-Exploitation Trade-off in General Intelligence.

Theoretical Context:
--------------------
In Exercises 1 and 2, we looked at the SUM (Upsilon) and the TEST (Environment). 
Now we look at the STRATEGY (The Agent's Policy).

Shane Legg defines a policy π as a mapping from History to Actions. 
In the real world, an agent doesn't know the 'shortest program' for the 
environment μ beforehand. It must DISCOVER it.

The Exploration Problem:
------------------------
This brings us to the fundamental trade-off of Reinforcement Learning:
- **Exploitation (Greed):** Using your current best model to maximize reward.
- **Exploration (Curiosity):** Doing something random to gather more data.

Universal Intelligence Insight:
-------------------------------
A universally intelligent agent cannot be 'purely greedy'. Why? 
Because if the environment is a complex pattern (high K), a greedy agent 
might get trapped in a "local cold spot" and never find the high-reward 
regions. 

However, if an agent explores *too much* (high epsilon), it becomes noisy. 
It might "stumble" upon the solution but fail to act on it consistently. 
In the Upsilon metric, this 'jitter' lowers the expected reward V_μ, and 
thus the intelligence score.

Your Task:
----------
1. Implement the `act()` method for the `EpsilonGreedyAgent`.
2. Understand how Epsilon (ε) acts as a "Information Filter".
3. Verify that the agent correctly switches between random and greedy modes.

Pedagogical Insight:
--------------------
Think of Epsilon as the "Softmax Temperature" from Day 2. 
- At ε = 1, the agent is a gas: random and high entropy.
- At ε = 0, the agent is a crystal: rigid and deterministic.
Intelligence requires a "phase transition" between the two.
"""

import numpy as np
import sys
import os
from typing import Any

# Ensure we can import the core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import Agent

# =============================================================================
# MASTERCLASS COMMENTARY: The Cost of Knowledge
# =============================================================================
# In Legg's thesis, the 'Speed' of learning is not the primary focus of 
# Upsilon, but the 'Capacity' to learn is. 
# An agent with ε=0 and a bad initial model will have an Upsilon of zero in 
# most environments because it never learns the rules.
#
# However, ε=0.1 is the "industry standard" for a reason—it provides enough 
# noise to escape local optima without destroying the agent's ability to 
# execute the patterns it has found.
# =============================================================================

class EpsilonGreedyAgent(Agent):
    """
    An agent that implements the ε-greedy strategy.
    
    It maintains a Q-Table (Internal Model) of state-action utilities.
    """
    def __init__(self, epsilon: float = 0.1, actions: int = 4):
        """
        Args:
            epsilon: The probability of taking a random action.
            actions: The number of possible moves in the environment.
        """
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}

    def act(self, observation: Any) -> int:
        """
        The decision-making heart of the agent.
        
        Instructional Logic:
        --------------------
        1. Stringify the observation to use as a key in our 'Brain' (Q-Table).
        2. If this is a new state, initialize its values to zero.
        3. Draw a random number between 0 and 1.
        4. If the number < ε: EXPLORE (Return a random action index).
        5. Else: EXPLOIT (Return the action with the maximum current 'score').
        
        Args:
            observation: The current 'View' of the environment.
            
        Returns:
            The chosen action index (0 to actions-1).
        """
        # Step 1: Pre-processing the view
        state = str(observation)
        
        # Step 2: Initialize memory if state is unknown
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions)
        
        # [YOUR CODE HERE]
        # Implement the branching logic for ε-greedy...
        
        # Default placeholder:
        chosen_action = 0
        
        return chosen_action

    def observe(self, observation: Any, reward: float, done: bool):
        """
        We omit the Bellman Update here to focus purely on the 
        Choice Architecture (the act method).
        """
        pass

# =============================================================================
# VERIFICATION SUITE: Testing the Choice Logic
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 27 EXERCISE 3: EXPLORATION STRATEGY")
    print("=" * 60)
    
    # CASE 1: The Perfectionist (ε = 0)
    print("\n[SCENARIO 1] Testing Pure Exploitation (ε = 0)...")
    agent_greedy = EpsilonGreedyAgent(epsilon=0.0)
    
    # Mock some 'learned' knowledge
    # Action 2 is the best (Value 50)
    agent_greedy.q_table["room_1"] = np.array([10, -5, 50, 0])
    
    decisions = [agent_greedy.act("room_1") for _ in range(100)]
    
    if all(d == 2 for d in decisions):
        print("  [PASS] Agent consistently chose the best action.")
    else:
        print(f"  [FAIL] Agent deviated from greedy choice! Logic error in epsilon check.")

    # CASE 2: The Chaos Monkey (ε = 1.0)
    print("\n[SCENARIO 2] Testing Pure Exploration (ε = 1.0)...")
    agent_random = EpsilonGreedyAgent(epsilon=1.0)
    agent_random.q_table["room_1"] = np.array([1000, 0, 0, 0]) # Big bias for action 0
    
    decisions = [agent_random.act("room_1") for _ in range(100)]
    unique_actions = set(decisions)
    
    print(f"  Actions taken over 100 steps: {unique_actions}")
    if len(unique_actions) > 1:
        print("  [PASS] Agent ignored its bias and explored the space.")
    else:
        print("  [FAIL] Agent stayed greedy despite ε=1.0.")

    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE: You understand the 'Epsilon' filter of Intelligence.")
    print("=" * 60)
