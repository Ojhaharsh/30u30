"""
Exercise 5: PPO vs. Vanilla Policy Gradient

Compare PPO-Clip against vanilla policy gradient (REINFORCE) to see
what the clipping actually buys you in terms of training stability.

Vanilla PG loss:
    L_PG = E[log_pi(a|s) * A_t]

This is equivalent to PPO with epsilon=infinity (no clipping).

The paper's claim (Section 3): without clipping, large gradient steps
can destroy the policy. With clipping, updates are bounded.

Reference: Schulman et al. (2017), Section 3
https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import ActorCritic, RolloutBuffer

try:
    import gymnasium as gym
except ImportError:
    import gym


def vanilla_pg_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    Vanilla policy gradient loss (no clipping).

    L_PG = -E[log_pi(a|s) * A_t]

    Note: log_probs_old is unused here (no ratio). This is the baseline
    that PPO improves upon.

    Args:
        log_probs_new: Log probs under current policy. Shape: (batch,)
        log_probs_old: Log probs under old policy. Shape: (batch,) [unused]
        advantages: Advantage estimates. Shape: (batch,)

    Returns:
        loss: Scalar loss (negative of the objective).
    """
    # TODO: Implement vanilla PG loss
    # It's just: -mean(log_probs_new * advantages)
    # Note: log_probs_old is intentionally unused (no importance sampling ratio)
    raise NotImplementedError("Implement vanilla_pg_loss")


def run_comparison(
    env_name: str = "CartPole-v1",
    n_iterations: int = 80,
    seed: int = 42,
) -> dict:
    """
    Train PPO-Clip and Vanilla PG on the same environment and compare.

    Returns:
        results: Dict with keys 'ppo' and 'vanilla_pg', each mapping to
                 a list of mean rewards per iteration.
    """
    results = {}

    for method in ["ppo", "vanilla_pg"]:
        print(f"Training {method}...")
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        network = ActorCritic(obs_dim, act_dim)
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
        buffer = RolloutBuffer(T=512, obs_dim=obs_dim, act_dim=1, gamma=0.99, lambda_=0.95)

        rewards = []

        for iteration in range(n_iterations):
            # Collect rollout
            obs, _ = env.reset()
            episode_rewards = []
            current_ep_reward = 0.0

            network.eval()
            with torch.no_grad():
                for t in range(512):
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    action, log_prob, value = network.get_action(obs_t)
                    next_obs, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    buffer.store(obs, action.item(), reward, float(done),
                                 log_prob.item(), value.item())
                    current_ep_reward += reward
                    obs = next_obs
                    if done:
                        episode_rewards.append(current_ep_reward)
                        current_ep_reward = 0.0
                        obs, _ = env.reset()

                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                _, _, last_val = network.get_action(obs_t)
                buffer.compute_advantages(last_val.item())

            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            rewards.append(mean_reward)

            # Update
            network.train()
            for epoch in range(4):
                for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.get_minibatches(64, torch.device("cpu")):
                    log_probs_new, values, entropy = network.evaluate(obs_b, act_b)

                    # Normalize advantages
                    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                    if method == "ppo":
                        # PPO-Clip loss (Equation 7)
                        ratio = torch.exp(log_probs_new - old_lp_b)
                        surr1 = ratio * adv_norm
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_norm
                        policy_loss = -torch.min(surr1, surr2).mean()
                    else:
                        # Vanilla PG loss (your implementation)
                        policy_loss = vanilla_pg_loss(log_probs_new, old_lp_b, adv_norm)

                    value_loss = F.mse_loss(values, ret_b)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + 1.0 * value_loss + 0.01 * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    optimizer.step()

        env.close()
        results[method] = rewards
        final_avg = np.mean(rewards[-10:])
        print(f"  {method}: final 10-iter avg = {final_avg:.1f}")

    return results


def analyze_comparison(results: dict):
    """Print comparison summary."""
    print()
    print("PPO vs. Vanilla Policy Gradient Comparison")
    print("-" * 50)
    for method, rewards in results.items():
        final_avg = np.mean(rewards[-10:])
        max_reward = max(rewards)
        print(f"  {method:15s}: final avg = {final_avg:6.1f}, max = {max_reward:6.1f}")

    print()
    print("Expected: PPO should be more stable (less variance in reward curve).")
    print("Vanilla PG may reach high rewards but then collapse.")


if __name__ == "__main__":
    print("Exercise 5: PPO vs. Vanilla Policy Gradient")
    print("First implement vanilla_pg_loss(), then run this script.")
    print()

    results = run_comparison(env_name="CartPole-v1", n_iterations=80)
    analyze_comparison(results)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        for method, rewards in results.items():
            ax.plot(rewards, label=method, alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("PPO-Clip vs. Vanilla Policy Gradient: CartPole-v1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
