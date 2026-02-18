"""
Exercise 3: Actor-Critic Network

Build the actor-critic network from scratch.

The network has:
- A shared backbone (2 hidden layers, tanh activation)
- An actor head: outputs logits over actions (for discrete action spaces)
- A critic head: outputs a scalar value estimate V(s)

Reference: PPO paper Section 3 and Algorithm 1
https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Shared actor-critic network for PPO (discrete action spaces).

    Args:
        obs_dim: Dimension of the observation space.
        act_dim: Number of discrete actions.
        hidden_dim: Width of hidden layers. Paper uses 64 for MuJoCo.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        # TODO: Build the shared backbone
        # Two linear layers with tanh activation
        # Input: obs_dim -> hidden_dim -> hidden_dim
        self.backbone = None  # replace with nn.Sequential(...)

        # TODO: Build the actor head
        # Linear layer: hidden_dim -> act_dim
        # Outputs logits (raw scores, not probabilities)
        self.actor_head = None  # replace with nn.Linear(...)

        # TODO: Build the critic head
        # Linear layer: hidden_dim -> 1
        # Outputs a scalar value estimate
        self.critic_head = None  # replace with nn.Linear(...)

    def get_action_and_value(self, obs: torch.Tensor):
        """
        Sample an action and return (action, log_prob, value).

        Args:
            obs: Observation tensor. Shape: (batch, obs_dim) or (obs_dim,)

        Returns:
            action: Sampled action. Shape: (batch,)
            log_prob: Log probability of the sampled action. Shape: (batch,)
            value: Value estimate. Shape: (batch,)
        """
        # TODO: Implement the forward pass
        # 1. Pass obs through backbone
        # 2. Compute logits from actor_head
        # 3. Create Categorical distribution from logits
        # 4. Sample action and compute log_prob
        # 5. Compute value from critic_head
        raise NotImplementedError("Implement get_action_and_value")

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate stored actions under the current policy.
        Used during the optimization phase.

        Args:
            obs: Observation tensor. Shape: (batch, obs_dim)
            actions: Stored actions. Shape: (batch,)

        Returns:
            log_probs: Log prob of stored actions under current policy. Shape: (batch,)
            values: Value estimates. Shape: (batch,)
            entropy: Policy entropy. Shape: (batch,)
        """
        # TODO: Implement evaluate_actions
        # Same as get_action_and_value but:
        # - Use the stored actions (don't sample new ones)
        # - Compute log_prob of the stored actions
        # - Also return entropy of the distribution
        raise NotImplementedError("Implement evaluate_actions")


def test_actor_critic():
    """Tests for the actor-critic network."""
    torch.manual_seed(42)

    obs_dim, act_dim = 4, 2  # CartPole dimensions
    net = ActorCriticNetwork(obs_dim, act_dim, hidden_dim=64)

    # Test 1: Output shapes are correct
    obs = torch.randn(8, obs_dim)  # batch of 8 observations
    action, log_prob, value = net.get_action_and_value(obs)

    assert action.shape == (8,), f"Test 1 failed: action shape {action.shape}"
    assert log_prob.shape == (8,), f"Test 1 failed: log_prob shape {log_prob.shape}"
    assert value.shape == (8,), f"Test 1 failed: value shape {value.shape}"
    print("[OK] Test 1: output shapes are correct")

    # Test 2: Actions are valid (in range [0, act_dim))
    assert action.min() >= 0 and action.max() < act_dim, (
        f"Test 2 failed: actions out of range: {action}"
    )
    print("[OK] Test 2: actions are in valid range")

    # Test 3: Log probabilities are negative (probabilities < 1)
    assert (log_prob <= 0).all(), f"Test 3 failed: log_probs should be <= 0"
    print("[OK] Test 3: log probabilities are non-positive")

    # Test 4: evaluate_actions returns the same log_prob for the same actions
    log_probs2, values2, entropy = net.evaluate_actions(obs, action)
    assert torch.allclose(log_prob, log_probs2, atol=1e-5), (
        "Test 4 failed: evaluate_actions should return same log_probs as get_action_and_value"
    )
    print("[OK] Test 4: evaluate_actions is consistent with get_action_and_value")

    # Test 5: Entropy is positive
    assert (entropy > 0).all(), f"Test 5 failed: entropy should be positive"
    print("[OK] Test 5: entropy is positive")

    print()
    print("All tests passed. Open solutions/solution_03_actor_critic.py to compare.")


if __name__ == "__main__":
    test_actor_critic()
