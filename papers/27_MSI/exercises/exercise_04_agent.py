"""
Day 27: Machine Super Intelligence | Exercise 4
==============================================

Goal: Implementing an Agent that embodies 'Intelligence = Compression'.

Theoretical Context:
--------------------
In Exercise 3, we optimized a strategy (Epsilon). Now, we move to the 
Deepest Pillar of Legg's thesis: the link between **Induction** and 
**Intelligence**.

Shane Legg (and his advisor Marcus Hutter) famously proposed that:
> "Intelligence is the ability to achieve goals in many environments."

To do this across *many* environments, an agent cannot rely on task-specific 
memorization. It must be able to "Induce" or "Compress" the rules of any 
arbitrary environment μ using the history of interactions.

The Solomonoff Proxy:
---------------------
The theoretical "Perfect Agent" (AIXI) uses **Solomonoff Induction**. 
It looks at everything it has ever seen and finds the *shortest computer 
program* that explains that data. It then uses that program to predict 
the future and choose its actions.

Since Finding the shortest program is non-computable, we use a proxy:
**Predictive Induction**. An agent that learns "If I am in state A and 
I take action X, I always see state B" is effectively compressing 
the environment into a state-transition graph.

Your Task:
----------
1. Implement the `PredictiveHistoryAgent`.
2. Unlike the RL agent (which learns values), this agent learns **Rules**.
3. Verify that the agent can "solve" a repeating pattern environment faster 
   than a random or naive greedy agent.

Pedagogical Insight:
--------------------
Recall Day 2 (LSTMs). The LSTM's "Hidden State" is effectively a compressed 
summary of the past. Here, our agent's `transitions` dictionary is its 
"Hidden State"—a formal map of the world's source code.
"""

import sys
import os
from typing import Any, List

# Ensure we can import the core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import Agent, PatternSequence

# =============================================================================
# MASTERCLASS COMMENTARY: The Engine of Induction
# =============================================================================
# Why is prediction better than just Q-learning?
# In a PatternSequence environment [0, 1, 2], the state is not just the 
# current number. The "Source Code" of the world is a loop.
#
# A predictive agent builds a map: (LastObservation, Action) -> NextObservation.
# Once it has seen (Current=0, Action=1 -> Next=1), it has discovered a piece 
# of the environment's "K-complexity". It can now predict the reward without 
# needing thousands of random trials.
# =============================================================================

class PredictiveHistoryAgent(Agent):
    """
    An agent that 'Compresses' experience into a transition map.
    
    It doesn't care about 'Values' (Q-scores) as much as it cares about 
    'Truth' (What happens next?).
    """
    def __init__(self, history_len: int = 10):
        """
        Args:
            history_len: How many past steps to track.
        """
        self.history = []
        self.transitions = {} # Mapping: current_state -> next_state prediction
        self.last_observation = None

    def act(self, observation: Any) -> int:
        """
        Decides the next action by querying the internal 'World Map'.
        
        Instructional Logic:
        --------------------
        1. Look at the current `observation`.
        2. Check if we have a rule in `self.transitions` for this state.
        3. If we DO: The 'Action' is the predicted next state (in this specific env).
        4. If we DON'T: Explore (take a random action or just guess).
        
        Args:
            observation: The number we just saw from the PatternSequence.
            
        Returns:
            The predicted next number in the sequence.
        """
        state = observation
        
        # [YOUR CODE HERE]
        # Querying the model: If transitions[state] exists, return it.
        # Otherwise, yield a default explore action...
        
        # Placeholder logic:
        predicted_action = 0
        
        return predicted_action

    def observe(self, observation: Any, reward: float, done: bool):
        """
        Updates the internal 'Model' of the environment's source code.
        
        Instructional Logic:
        --------------------
        1. If we have a `self.last_observation`:
           We now know that (last_observation -> current_observation) is a rule.
        2. Update `self.transitions[last_observation] = current_observation`.
        3. Update `self.last_observation` to the current one.
        """
        # [YOUR CODE HERE]
        
        # Always track the history
        self.last_observation = observation
        self.history.append(observation)


# =============================================================================
# VERIFICATION SUITE: Pattern Recognition Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 27 EXERCISE 4: THE INDUCTION ENGINE")
    print("=" * 60)
    
    # Environment: A simple repeating loop [0, 1, 2]
    # K is low!
    pattern = [0, 1, 2]
    print(f"\n[STEP 1] Initializing PatternSequence({pattern})...")
    env = PatternSequence(pattern)
    agent = PredictiveHistoryAgent()
    
    obs = env.reset()
    total_reward = 0
    steps = 20
    
    print(f"\n[STEP 2] Simulating {steps} steps of interaction...")
    
    for i in range(steps):
        action = agent.act(obs)
        prev_obs = obs
        obs, rew, done = env.step(action)
        agent.observe(obs, rew, done)
        total_reward += rew
        
        # Progress log matching Day 2's iterative reporting:
        if i < 5:
            print(f"  Step {i}: Saw {prev_obs}, Predicted {action}, Got {obs} | Reward: {rew}")

    print(f"\n  Final Total Reward: {total_reward} / {steps}")
    
    # A perfect predictive agent should miss the first few steps while learning
    # length of pattern = 3. Learning 0->1, 1->2, 2->0 takes 3-4 steps.
    # threshold = 20 - 4 = 16.
    if total_reward >= 15:
        print("  [PASS] Agent induced the environment's program successfully.")
    else:
        print(f"  [FAIL] Agent failed to recognize the pattern. Reward: {total_reward}")

    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE: You have implemented a Solomonoff-lite Induction.")
    print("=" * 60)
