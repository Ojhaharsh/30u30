"""
implementation.py - Proximal Policy Optimization (PPO-Clip) in PyTorch

A complete, educational implementation of PPO-Clip from:
    Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
    https://arxiv.org/abs/1707.06347

Key Components:
    1. Actor-Critic Network (shared backbone, separate heads)
    2. Generalized Advantage Estimation (GAE, lambda=0.95)
    3. Clipped Surrogate Objective (Equation 7 from the paper)
    4. Full PPO Loss: L_CLIP + c1*L_VF - c2*Entropy (Equation 9)
    5. Training loop with multiple epochs per data batch

The implementation follows the paper's MuJoCo hyperparameters (Table 3)
as the default, with CartPole-v1 as the demo environment.

Author: 30u30 Project
Reference: https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional
import argparse


# ============================================================================
# SECTION 1: ACTOR-CRITIC NETWORK
# ============================================================================
# PPO uses an actor-critic architecture. The actor outputs a policy
# (probability distribution over actions), and the critic outputs a
# value estimate V(s). They can share a backbone (as here) or be
# completely separate networks.
# ============================================================================

class ActorCritic(nn.Module):
    """
    Shared actor-critic network for PPO.

    Architecture: two hidden layers (64 units each, tanh activation)
    shared between actor and critic, with separate output heads.

    The paper uses tanh activations and orthogonal initialization
    for MuJoCo tasks. For Atari, a CNN backbone replaces the MLP.

    Args:
        obs_dim: Dimension of the observation space.
        act_dim: Number of discrete actions (or action dimension for continuous).
        hidden_dim: Width of hidden layers (paper uses 64).
        continuous: If True, outputs Gaussian policy for continuous actions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous

        # Shared backbone: two hidden layers with tanh activation.
        # Tanh is preferred over ReLU for RL because it bounds activations,
        # which helps with the value function estimation.
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs action logits (discrete) or mean (continuous)
        self.actor_head = nn.Linear(hidden_dim, act_dim)

        # Critic head: outputs scalar value estimate V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

        if continuous:
            # Log standard deviation: learned parameter, not state-dependent.
            # The paper uses a state-independent log_std for MuJoCo tasks.
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Orthogonal initialization: the paper's implementation uses this.
        # It helps with gradient flow in deep networks.
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization with gain=sqrt(2) for hidden layers."""
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass: returns (action distribution, value estimate).

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            dist: Action distribution (Categorical or Normal).
            value: Value estimate of shape (batch, 1).
        """
        features = self.backbone(obs)
        value = self.critic_head(features)

        if self.continuous:
            mean = self.actor_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
        else:
            logits = self.actor_head(features)
            dist = Categorical(logits=logits)

        return dist, value

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return (action, log_prob, value).
        Used during data collection (rollout phase).
        """
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)  # sum over action dimensions
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored actions under the current policy.
        Used during the optimization phase (K epochs).

        Returns:
            log_probs: Log probability of stored actions under current policy.
            values: Current value estimates.
            entropy: Policy entropy (for the entropy bonus in Equation 9).
        """
        dist, value = self.forward(obs)
        log_probs = dist.log_prob(actions)
        if self.continuous:
            log_probs = log_probs.sum(dim=-1)
        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        return log_probs, value.squeeze(-1), entropy


# ============================================================================
# SECTION 2: ROLLOUT BUFFER
# ============================================================================
# Stores T timesteps of experience (obs, actions, rewards, dones, log_probs,
# values). After collection, computes GAE advantages and returns.
# ============================================================================

class RolloutBuffer:
    """
    Stores one iteration of experience for PPO.

    The buffer holds T timesteps from a single actor. After collection,
    compute_advantages() is called once to compute GAE advantages and
    discounted returns. The buffer is then split into minibatches for
    K epochs of optimization.

    Args:
        T: Number of timesteps to collect per iteration.
        obs_dim: Observation dimension.
        act_dim: Action dimension (1 for discrete).
        gamma: Discount factor (paper default: 0.99).
        lambda_: GAE lambda (paper default: 0.95).
    """

    def __init__(self, T: int, obs_dim: int, act_dim: int, gamma: float = 0.99, lambda_: float = 0.95):
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_

        self.obs = np.zeros((T, obs_dim), dtype=np.float32)
        self.actions = np.zeros(T, dtype=np.int64)
        self.rewards = np.zeros(T, dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.float32)
        self.log_probs = np.zeros(T, dtype=np.float32)
        self.values = np.zeros(T + 1, dtype=np.float32)  # +1 for bootstrap value

        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns = np.zeros(T, dtype=np.float32)

        self.ptr = 0

    def store(self, obs, action, reward, done, log_prob, value):
        """Store one timestep of experience."""
        assert self.ptr < self.T, "Buffer is full."
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_advantages(self, last_value: float):
        """
        Compute GAE advantages and discounted returns.

        GAE (Generalized Advantage Estimation, Schulman et al. 2015b):
            A_t = sum_l (gamma * lambda)^l * delta_{t+l}
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

        This is a backward pass through the stored timesteps.
        The (1 - done) term zeros out the bootstrap value at episode boundaries.

        Args:
            last_value: V(s_{T+1}), the bootstrap value for the last state.
        """
        self.values[self.T] = last_value
        gae = 0.0
        for t in reversed(range(self.T)):
            # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            # (1 - done) masks out the next value at episode boundaries
            delta = (
                self.rewards[t]
                + self.gamma * self.values[t + 1] * (1.0 - self.dones[t])
                - self.values[t]
            )
            # GAE: accumulate discounted TD residuals
            gae = delta + self.gamma * self.lambda_ * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae

        # Returns = advantages + values (used as targets for the value function)
        self.returns = self.advantages + self.values[:self.T]
        self.ptr = 0

    def get_minibatches(self, minibatch_size: int, device: torch.device):
        """
        Yield shuffled minibatches for K epochs of optimization.

        The paper shuffles the data at the start of each epoch and splits
        into minibatches of size M (default 64 for MuJoCo).
        """
        indices = np.random.permutation(self.T)
        for start in range(0, self.T, minibatch_size):
            idx = indices[start:start + minibatch_size]
            yield (
                torch.FloatTensor(self.obs[idx]).to(device),
                torch.LongTensor(self.actions[idx]).to(device),
                torch.FloatTensor(self.log_probs[idx]).to(device),
                torch.FloatTensor(self.advantages[idx]).to(device),
                torch.FloatTensor(self.returns[idx]).to(device),
            )


# ============================================================================
# SECTION 3: PPO LOSS (THE CORE CONTRIBUTION)
# ============================================================================
# Implements Equation 7 (clipped surrogate) and Equation 9 (full objective)
# from the paper.
# ============================================================================

def ppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    epsilon: float = 0.2,
    c1: float = 1.0,
    c2: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the full PPO loss (Equation 9 from the paper).

    L = L_CLIP - c1 * L_VF + c2 * S[pi]

    Args:
        log_probs_new: Log probs of stored actions under current policy.
        log_probs_old: Log probs of stored actions under old policy (detached).
        advantages: GAE advantage estimates (normalized within minibatch).
        values: Current value estimates V(s_t).
        returns: Discounted returns (targets for value function).
        entropy: Policy entropy.
        epsilon: Clip range (paper default: 0.2).
        c1: Value function loss coefficient (paper default: 1.0).
        c2: Entropy bonus coefficient (paper default: 0.01 for Atari, 0 for MuJoCo).

    Returns:
        loss: Scalar loss to minimize (note: we minimize -L, so we maximize L).
        info: Dict with individual loss components for logging.
    """
    # Normalize advantages within this minibatch.
    # Not in the paper's equations but standard practice and important for stability.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Probability ratio: r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
    # Computed in log space for numerical stability.
    # log_probs_old must be detached (computed from old policy, not current).
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Clipped surrogate objective (Equation 7):
    # L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()  # negative: we minimize

    # Value function loss: MSE between current value estimates and returns
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus: encourages exploration by penalizing low-entropy policies
    entropy_loss = -entropy.mean()  # negative: we want to maximize entropy

    # Full objective (Equation 9): L = L_CLIP - c1*L_VF + c2*S
    # We minimize: policy_loss + c1*value_loss + c2*entropy_loss
    # (signs already flipped above)
    total_loss = policy_loss + c1 * value_loss + c2 * entropy_loss

    # Clipping fraction: fraction of timesteps where the clip was active.
    # Useful diagnostic: should be non-zero but not too high (5-30%).
    clip_fraction = ((ratio - 1.0).abs() > epsilon).float().mean().item()

    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": -entropy_loss.item(),
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean().item(),
    }

    return total_loss, info


# ============================================================================
# SECTION 4: PPO AGENT
# ============================================================================

class PPOAgent:
    """
    PPO-Clip agent following Algorithm 1 from the paper.

    The training loop:
        for each iteration:
            1. Collect T timesteps using current policy (rollout)
            2. Compute GAE advantages
            3. For K epochs:
                a. Split data into minibatches of size M
                b. Compute PPO loss on each minibatch
                c. Update policy with gradient step

    Args:
        obs_dim: Observation space dimension.
        act_dim: Action space dimension.
        T: Timesteps per iteration (paper: 2048 for MuJoCo).
        K: Epochs per iteration (paper: 10 for MuJoCo).
        M: Minibatch size (paper: 64 for MuJoCo).
        epsilon: Clip range (paper: 0.2).
        gamma: Discount factor (paper: 0.99).
        lambda_: GAE lambda (paper: 0.95).
        lr: Learning rate (paper: 3e-4).
        c1: Value loss coefficient (paper: 1.0).
        c2: Entropy coefficient (paper: 0.01 for Atari, 0 for MuJoCo).
        max_grad_norm: Gradient clipping (paper: 0.5).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        T: int = 2048,
        K: int = 10,
        M: int = 64,
        epsilon: float = 0.2,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        lr: float = 3e-4,
        c1: float = 1.0,
        c2: float = 0.0,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.T = T
        self.K = K
        self.M = M
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.c1 = c1
        self.c2 = c2
        self.device = torch.device(device)

        self.network = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(T, obs_dim, 1, gamma, lambda_)

    def collect_rollout(self, env) -> float:
        """
        Collect T timesteps of experience using the current policy.

        Returns the mean episode reward over completed episodes.
        """
        obs, _ = env.reset()
        episode_rewards = []
        current_episode_reward = 0.0

        self.network.eval()
        with torch.no_grad():
            for t in range(self.T):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, log_prob, value = self.network.get_action(obs_tensor)

                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                self.buffer.store(obs, action_np, reward, float(done), log_prob_np, value_np)
                current_episode_reward += reward
                obs = next_obs

                if done:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0.0
                    obs, _ = env.reset()

            # Bootstrap value for the last state
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, last_value = self.network.get_action(obs_tensor)
            last_value_np = last_value.cpu().numpy()[0]

        self.buffer.compute_advantages(last_value_np)
        return np.mean(episode_rewards) if episode_rewards else 0.0

    def update(self) -> dict:
        """
        Run K epochs of PPO optimization on the collected data.

        Returns aggregated loss statistics for logging.
        """
        self.network.train()
        all_info = []

        for epoch in range(self.K):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in self.buffer.get_minibatches(
                self.M, self.device
            ):
                # Evaluate stored actions under the CURRENT policy.
                # This is the key: log_probs_new changes each epoch as the policy updates.
                log_probs_new, values, entropy = self.network.evaluate(obs_b, act_b)

                loss, info = ppo_loss(
                    log_probs_new=log_probs_new,
                    log_probs_old=old_lp_b,  # detached: computed from old policy
                    advantages=adv_b,
                    values=values,
                    returns=ret_b,
                    entropy=entropy,
                    epsilon=self.epsilon,
                    c1=self.c1,
                    c2=self.c2,
                )

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: the paper clips to max_norm=0.5
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                all_info.append(info)

        # Aggregate stats across all minibatches and epochs
        return {k: np.mean([d[k] for d in all_info]) for k in all_info[0]}


# ============================================================================
# SECTION 5: DEMO
# ============================================================================

def run_demo():
    """
    Demo: train PPO on CartPole-v1 for 50 iterations.

    CartPole-v1 is considered solved when the average reward over
    100 consecutive episodes exceeds 475.
    """
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    print("PPO-Clip Demo: CartPole-v1")
    print("Paper: Schulman et al. (2017) - https://arxiv.org/abs/1707.06347")
    print("-" * 60)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        T=512,       # smaller T for faster demo
        K=4,
        M=64,
        epsilon=0.2,
        gamma=0.99,
        lambda_=0.95,
        lr=3e-4,
        c1=1.0,
        c2=0.01,     # entropy bonus for discrete actions
    )

    print(f"Network: {sum(p.numel() for p in agent.network.parameters())} parameters")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"T={agent.T}, K={agent.K}, M={agent.M}, epsilon={agent.epsilon}")
    print()

    for iteration in range(50):
        mean_reward = agent.collect_rollout(env)
        stats = agent.update()

        if (iteration + 1) % 5 == 0:
            print(
                f"Iter {iteration+1:3d} | "
                f"reward={mean_reward:6.1f} | "
                f"policy_loss={stats['policy_loss']:+.4f} | "
                f"value_loss={stats['value_loss']:.4f} | "
                f"entropy={stats['entropy']:.4f} | "
                f"clip_frac={stats['clip_fraction']:.3f}"
            )

    env.close()
    print()
    print("Done. CartPole-v1 is considered solved at mean reward >= 475.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-Clip implementation (Day 29)")
    parser.add_argument("--demo", action="store_true", help="Run CartPole demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        print("Run with --demo to see the CartPole training demo.")
        print("Or import PPOAgent, ActorCritic, ppo_loss for your own use.")
