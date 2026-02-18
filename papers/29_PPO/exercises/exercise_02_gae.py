"""
Exercise 2: Generalized Advantage Estimation (GAE)

Implement GAE from Schulman et al. (2015b), which is the advantage estimator
used by PPO (cited in the PPO paper, Section 4).

GAE formula:
    A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

This is computed via a backward recurrence:
    gae = 0
    for t in reversed(range(T)):
        delta = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
        gae = delta + gamma * lambda * (1 - done_t) * gae
        A_t = gae

Reference: Schulman et al. (2015b) - https://arxiv.org/abs/1506.02438
           PPO paper Section 4 - https://arxiv.org/abs/1707.06347
"""

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """
    Compute GAE advantage estimates.

    Args:
        rewards: Array of rewards. Shape: (T,)
        values: Array of value estimates INCLUDING the bootstrap value.
                Shape: (T+1,). values[T] is V(s_{T+1}).
        dones: Array of episode termination flags. Shape: (T,)
               1.0 if the episode ended at step t, 0.0 otherwise.
        gamma: Discount factor. PPO paper default: 0.99.
        lambda_: GAE lambda. PPO paper default: 0.95.
                 lambda=0: one-step TD (low variance, high bias)
                 lambda=1: Monte Carlo (high variance, low bias)

    Returns:
        advantages: GAE advantage estimates. Shape: (T,)

    Hints:
        - Iterate backward: for t in reversed(range(T))
        - delta_t = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        - gae = delta_t + gamma * lambda_ * (1 - dones[t]) * gae
        - The (1 - dones[t]) term zeros out the bootstrap at episode boundaries
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)

    # TODO: Implement GAE using the backward recurrence
    # gae = 0.0
    # for t in reversed(range(T)):
    #     ...
    raise NotImplementedError("Implement compute_gae")


def test_gae():
    """Tests for GAE computation."""
    np.random.seed(42)

    # Test 1: lambda=0 should give one-step TD advantage
    # A_t = r_t + gamma * V_{t+1} - V_t (when lambda=0 and no episode ends)
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.5, 0.5, 0.5])  # T+1 values
    dones = np.array([0.0, 0.0, 0.0])
    gamma, lambda_ = 0.99, 0.0

    adv = compute_gae(rewards, values, dones, gamma, lambda_)
    expected = rewards + gamma * values[1:] - values[:-1]
    assert np.allclose(adv, expected, atol=1e-5), (
        f"Test 1 failed: expected {expected}, got {adv}"
    )
    print("[OK] Test 1: lambda=0 gives one-step TD advantage")

    # Test 2: At episode boundaries (done=1), the bootstrap value should be zeroed out.
    # If done=1 at t=1, then A_1 = r_1 + 0 - V_1 (no future value)
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.5, 0.5, 0.5])
    dones = np.array([0.0, 1.0, 0.0])  # episode ends at t=1
    gamma, lambda_ = 0.99, 0.0

    adv = compute_gae(rewards, values, dones, gamma, lambda_)
    # At t=1 (done=1): delta = r_1 + gamma * V_2 * (1-1) - V_1 = 1.0 - 0.5 = 0.5
    assert abs(adv[1] - 0.5) < 1e-5, f"Test 2 failed: adv[1]={adv[1]:.4f}, expected 0.5"
    print("[OK] Test 2: episode boundary zeroes out bootstrap value")

    # Test 3: lambda=1 should give Monte Carlo-like returns minus baseline
    # For a single episode with no discounting (gamma=1, lambda=1):
    # A_0 = (r_0 + r_1 + r_2 + V_3) - V_0
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.0, 0.0, 0.0, 0.0])  # zero baseline
    dones = np.array([0.0, 0.0, 0.0])
    gamma, lambda_ = 1.0, 1.0

    adv = compute_gae(rewards, values, dones, gamma, lambda_)
    # With zero baseline: A_t = sum of future rewards
    expected_0 = 3.0  # r_0 + r_1 + r_2
    expected_1 = 2.0  # r_1 + r_2
    expected_2 = 1.0  # r_2
    assert abs(adv[0] - expected_0) < 1e-5, f"Test 3 failed: adv[0]={adv[0]:.4f}"
    assert abs(adv[1] - expected_1) < 1e-5, f"Test 3 failed: adv[1]={adv[1]:.4f}"
    assert abs(adv[2] - expected_2) < 1e-5, f"Test 3 failed: adv[2]={adv[2]:.4f}"
    print("[OK] Test 3: lambda=1 gives Monte Carlo returns (zero baseline)")

    print()
    print("All tests passed. Open solutions/solution_02_gae.py to compare.")


if __name__ == "__main__":
    test_gae()
