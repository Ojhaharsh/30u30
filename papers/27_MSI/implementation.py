"""
implementation.py - Universal Intelligence Framework in NumPy

A complete, educational implementation of Shane Legg's Universal Intelligence
formalizing the Agent-Environment loop and the Upsilon metric.

Key Components:
1. The Agent-Environment Loop (A, O, R)
2. Kolmogorov Complexity Proxies (K)
3. The Universal Prior (2^-K)
4. Reference Agents (Random, RL, and Predictive)

Reference: Shane Legg (2008) - http://www.vetta.org/documents/Machine_Super_Intelligence.pdf
"""

This "from-scratch" approach mirrors the pedagogical depth of Day 2 (LSTMs),
focusing on the mathematical interaction between memory and general induction.

Author: 30u30 Project
License: MIT
"""

import numpy as np
import abc
from typing import List, Dict, Tuple, Any


def sigmoid(x):
    """
    Sigmoid activation function.
    
    In the context of RL agents, sigmoid is often used to map values 
    into a probability space (0, 1), similar to how gates work in Day 2.
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """
    Softmax activation (numerically stable).
    
    Converts a vector of scores into a probability distribution.
    Essential for agents to choose actions in a stochastic policy.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class Environment(abc.ABC):
    """
    Abstract Environment interface following Shane Legg's (A, O, R) framework.
    
    Legg treats the environment (μ) as a 'computable' program. 
    If you could write the world as a piece of code, its length would be 
    the Kolmogorov complexity K(μ).
    
    Theoretical Note:
    -----------------
    In the limit of AIXI, every possible program is an environment. 
    Here, we implement specific classes as proxies for those programs.
    """
    def __init__(self, complexity: int):
        # K(mu): The 'shortest program' that generates this task.
        # This is the "Occam's Razor" metric.
        self.complexity = complexity  

    @abc.abstractmethod
    def reset(self) -> Any:
        """Resets the environment state for a new episode."""
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Processes an action and returns (Observation, Reward, Done).
        
        This is the fundamental interaction primitive of the 
        Agent-Environment loop.
        """
        pass


class Agent(abc.ABC):
    """
    Abstract Agent interface (Policy π).
    
    In Legg's thesis, a policy is a function from History (H) to 
    an Action (A). History is the sequence of all past observations, 
    actions, and rewards.
    """
    @abc.abstractmethod
    def act(self, observation: Any) -> int:
        """Chooses an action based on the current observation and history."""
        pass

    @abc.abstractmethod
    def observe(self, observation: Any, reward: float, done: bool):
        """Allows the agent to learn from the consequences of its actions."""
        pass

# =============================================================================
# 2. UNIVERSAL ENVIRONMENTS Suite
# =============================================================================

# =============================================================================
# 2. UNIVERSAL ENVIRONMENTS Suite
# =============================================================================

class GridWorld(Environment):
    """
    A 2D navigation task where the agent must reach a target coordinate.
    
    Complexity (K):
    ---------------
    We represent K as the grid 'distance' or 'size'. A larger grid requires 
    a more complex program/logic to navigate efficiently.
    """
    def __init__(self, size: int = 5):
        super().__init__(complexity=size)
        self.size = size
        self.pos = [0, 0]
        self.target = [size - 1, size - 1]

    def reset(self):
        self.pos = [0, 0]
        return tuple(self.pos)

    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Action Mapping: 0:up, 1:down, 2:left, 3:right
        """
        if action == 0 and self.pos[1] < self.size - 1: self.pos[1] += 1
        elif action == 1 and self.pos[1] > 0: self.pos[1] -= 1
        elif action == 2 and self.pos[0] > 0: self.pos[0] -= 1
        elif action == 3 and self.pos[0] < self.size - 1: self.pos[0] += 1
        
        done = self.pos == self.target
        # Pedagogical Note: We use a small step penalty to encourage 
        # the agent to find the shortest 'program' for navigation.
        reward = 1.0 if done else -0.01 
        return tuple(self.pos), reward, done


class PatternSequence(Environment):
    """
    A logic induction task. The agent must predict the next number.
    
    Complexity (K):
    ---------------
    We define complexity as the length of the pattern description. 
    A repeating pattern like [0, 1] is simpler than a long sequence like [0, 1, 2, 3, 4].
    """
    def __init__(self, pattern: List[int]):
        super().__init__(complexity=len(pattern) * 2)
        self.pattern = pattern
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self.pattern[0]

    def step(self, action: int) -> Tuple[Any, float, bool]:
        # The 'Truth' of the environment is the next element in the pattern
        correct_next = self.pattern[(self.idx + 1) % len(self.pattern)]
        correct = action == correct_next
        reward = 1.0 if correct else 0.0
        
        self.idx += 1
        done = self.idx >= 10 # 10 steps of induction
        return self.pattern[self.idx % len(self.pattern)], reward, done


# =============================================================================
# 3. UNIVERSAL INTELLIGENCE MEASURE (Upsilon)
# =============================================================================

class UniversalIntelligenceMeasure:
    """
    Calculates Υ(π) — The Universal Intelligence Score.
    
    This class implements the summative logic of Shane Legg's theory. 
    It iterates through a suite of environments, weights the performance 
    by 2^-K, and produces the final scalar metric.
    """
    def __init__(self, environments: List[Environment]):
        self.environments = environments

    def evaluate(self, agent: Agent, episodes: int = 10) -> Dict[str, Any]:
        """
        Performs the universal benchmarking.
        
        Mathematical Process:
        ---------------------
        1. Calculate Weight: w = 2^(-complexity)
        2. Gather Performance: V = mean(episode rewards)
        3. Weighted Sum: Upsilon = Σ (w * V)
        """
        results = []
        total_upsilon = 0.0
        total_weight = 0.0

        for env in self.environments:
            # Universal Prior Weight: 2^-K(mu) 
            # This is the "Occam's Prior" that favors simplicity.
            weight = 2.0**(-env.complexity)
            
            episode_rewards = []
            for _ in range(episodes):
                obs = env.reset()
                done = False
                total_reward = 0.0
                while not done:
                    action = agent.act(obs)
                    obs, reward, done = env.step(action)
                    agent.observe(obs, reward, done)
                    total_reward += reward
                episode_rewards.append(total_reward)
            
            # Expected Value V_mu^pi (normalized)
            v_mu = max(0, np.mean(episode_rewards)) 
            total_upsilon += weight * v_mu
            total_weight += weight
            
            results.append({
                "env": type(env).__name__,
                "complexity": env.complexity,
                "weight": weight,
                "expected_reward": v_mu
            })

        return {
            "upsilon_raw": total_upsilon,
            "upsilon_normalized": total_upsilon / total_weight if total_weight > 0 else 0,
            "details": results
        }

# =============================================================================
# 4. REFERENCE AGENTS
# =============================================================================

# =============================================================================
# 4. REFERENCE AGENTS
# =============================================================================

class RandomAgent(Agent):
    """
    The baseline of intelligence (Υ ≈ 0).
    
    This agent acts without memory, logic, or induction. It represents the 
    'Null Hypothesis' of intelligence. If a learner cannot outperform this 
    baseline across the spectrum, it is not universally intelligent.
    """
    def act(self, observation: Any) -> int:
        return np.random.randint(0, 4)

    def observe(self, observation: Any, reward: float, done: bool):
        # The random agent lacks the capacity to integrate feedback.
        pass


class SimpleRLAgent(Agent):
    """
    A Gradient-based Learner (Q-Table Proxy).
    
    This represents 'Specialized Intelligence'. It is very good at learning 
    local state-action correlations within a single environment, but 
    struggles with 'Out-of-Distribution' tasks because its memory is 
    finite and task-specific.
    """
    def __init__(self, actions: int = 4):
        self.q_table = {}
        self.actions = actions
        self.lr = 0.1
        self.gamma = 0.9
        self.last_state = None
        self.last_action = None

    def act(self, observation: Any) -> int:
        state = str(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions)
        
        # Epsilon-greedy: Exploration is the fundamental cost of gathering information.
        if np.random.random() < 0.1:
            action = np.random.randint(0, self.actions)
        else:
            action = np.argmax(self.q_table[state])
        
        self.last_state = state
        self.last_action = action
        return action

    def observe(self, observation: Any, reward: float, done: bool):
        state = str(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions)
            
        if self.last_state is not None:
            # The Bellman Equation: Local Temporal Difference logic.
            # This is 'Intelligent' in a narrow sense—it optimizes a reward signal.
            best_next = np.max(self.q_table[state])
            self.q_table[self.last_state][self.last_action] += self.lr * (
                reward + self.gamma * best_next - self.q_table[self.last_state][self.last_action]
            )


class PredictiveAgent(Agent):
    """
    [Our Addition: Pedagogical Proxy for Solomonoff Induction]
    
    In Shane Legg's theory, the most intelligent agent is the one that 
    attempts to COMPRESS the history of observations into the shortest 
    possible internal model.
    
    Instead of just maximizing reward, this agent tries to find the 
    'simplest program' that explains the environment's transitions.
    """
    def __init__(self, actions: int = 4):
        self.history = []
        self.model = {} # transition model: (past_obs, action) -> next_obs
        self.reward_map = {} # (past_obs, action) -> expected_reward
        self.actions = actions
        self.last_obs = None
        self.last_action = None

    def act(self, observation: Any) -> int:
        state = str(observation)
        
        # If we have an internal model for this state, pick the predicted best action.
        best_action = 0
        max_pred_reward = -float('inf')
        
        for a in range(self.actions):
            pred_reward = self.reward_map.get((state, a), 0)
            if pred_reward > max_pred_reward:
                max_pred_reward = pred_reward
                best_action = a
        
        # Exploration: Vital for discovering the environment's 'program'.
        if np.random.random() < 0.1:
            return np.random.randint(0, self.actions)
            
        self.last_obs = state
        self.last_action = best_action
        return best_action

    def observe(self, observation: Any, reward: float, done: bool):
        next_state = str(observation)
        if self.last_obs is not None:
            # Model Update: Attempting to 'Induce' the rules of μ.
            # This mirrors the core of the MSI thesis.
            self.model[(self.last_obs, self.last_action)] = next_state
            
            # Predictive Reward Update
            old_r = self.reward_map.get((self.last_obs, self.last_action), 0)
            self.reward_map[(self.last_obs, self.last_action)] = 0.9 * old_r + 0.1 * reward
        
        if done:
            self.last_obs = None
            self.last_action = None


if __name__ == "__main__":
    print("Machine Super Intelligence | Framework Benchmark")
    print("=" * 60)
    print("\nThis implementation formalizes Shane Legg's thesis.")
    print("Use train_minimal.py to run the evaluation.")
    print("\nCore Principles Verified:")
    print("  - Universal Prior (2^-K) implementation")
    print("  - Non-anthropocentric Agent Benchmarking")
    print("  - Logic vs. Entropy Spectrum")
