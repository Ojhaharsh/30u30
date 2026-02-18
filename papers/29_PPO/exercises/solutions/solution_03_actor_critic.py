"""
Solution 3: Actor-Critic Network

Reference: PPO paper Section 3 and Algorithm 1
https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Shared actor-critic network for PPO (discrete action spaces).

    Architecture follows the paper: two hidden layers (64 units, tanh),
    shared backbone with separate actor and critic heads.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Shared backbone: two hidden layers with tanh activation.
        # Tanh is preferred over ReLU for RL because it bounds activations,
        # which helps with value function estimation stability.
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs logits over actions.
        # Small initialization (gain=0.01) keeps the initial policy close to uniform.
        self.actor_head = nn.Linear(hidden_dim, act_dim)

        # Critic head: outputs a scalar value estimate V(s).
        self.critic_head = nn.Linear(hidden_dim, 1)

    def get_action_and_value(self, obs: torch.Tensor):
        """
        Sample an action and return (action, log_prob, value).
        """
        features = self.backbone(obs)

        # Actor: compute action distribution
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic: compute value estimate
        value = self.critic_head(features).squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate stored actions under the current policy.

        This is called during the K optimization epochs. The key difference
        from get_action_and_value: we use the stored actions (not new samples),
        so we can compute the ratio r_t = pi_new / pi_old.
        """
        features = self.backbone(obs)

        logits = self.actor_head(features)
        dist = Categorical(logits=logits)

        # Evaluate the stored actions (not new samples)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        value = self.critic_head(features).squeeze(-1)

        return log_probs, value, entropy


# Run tests
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from exercises.exercise_03_actor_critic import test_actor_critic
    import exercises.exercise_03_actor_critic as ex
    ex.ActorCriticNetwork = ActorCriticNetwork
    test_actor_critic()
