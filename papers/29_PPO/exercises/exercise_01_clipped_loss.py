"""
Exercise 1: Clipped Surrogate Objective

Implement the PPO-Clip objective from Equation 7 of the paper:

    L_CLIP(theta) = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

where r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

Reference: Schulman et al. (2017), Equation 7
https://arxiv.org/abs/1707.06347
"""

import torch
import numpy as np


def clipped_surrogate_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Compute the clipped surrogate objective (Equation 7 from the paper).

    Args:
        log_probs_new: Log probabilities of actions under the CURRENT policy.
                       Shape: (batch_size,)
        log_probs_old: Log probabilities of actions under the OLD policy.
                       Shape: (batch_size,). Must be detached (no gradient).
        advantages: Advantage estimates. Shape: (batch_size,)
        epsilon: Clip range. Paper default: 0.2.

    Returns:
        loss: Scalar loss (negative of the objective, for minimization).

    Hints:
        - Compute the ratio in log space: ratio = exp(log_new - log_old)
        - The clip is: torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        - Take the min of clipped and unclipped: torch.min(surr1, surr2)
        - Return the NEGATIVE mean (we minimize, but the objective is maximized)
    """
    # TODO: Implement the clipped surrogate loss
    # Step 1: Compute probability ratio r_t = pi_new / pi_old
    ratio = None  # replace with your implementation

    # Step 2: Compute unclipped surrogate: r_t * A_t
    surr1 = None  # replace with your implementation

    # Step 3: Compute clipped surrogate: clip(r_t, 1-eps, 1+eps) * A_t
    surr2 = None  # replace with your implementation

    # Step 4: Take the minimum (pessimistic lower bound)
    # Step 5: Return negative mean (we minimize, objective is maximized)
    raise NotImplementedError("Implement clipped_surrogate_loss")


def test_clipped_surrogate_loss():
    """Tests for the clipped surrogate loss."""
    torch.manual_seed(42)

    # Test 1: When ratio = 1 (new policy = old policy), loss should equal
    # the vanilla policy gradient loss: -mean(advantages)
    log_probs = torch.zeros(100)
    advantages = torch.randn(100)
    loss = clipped_surrogate_loss(log_probs, log_probs, advantages, epsilon=0.2)
    expected = -advantages.mean()
    assert abs(loss.item() - expected.item()) < 1e-5, (
        f"Test 1 failed: expected {expected.item():.4f}, got {loss.item():.4f}"
    )
    print("[OK] Test 1: ratio=1 gives vanilla PG loss")

    # Test 2: When advantage is positive and ratio > 1+epsilon,
    # the clip should activate and the loss should be capped.
    # Specifically: loss = -(1 + epsilon) * advantage (for positive advantage)
    log_probs_new = torch.tensor([2.0])  # ratio = exp(2) >> 1+epsilon
    log_probs_old = torch.tensor([0.0])
    advantages = torch.tensor([1.0])  # positive advantage
    loss = clipped_surrogate_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2)
    expected = -(1.0 + 0.2) * 1.0  # clipped at 1+epsilon
    assert abs(loss.item() - expected) < 1e-5, (
        f"Test 2 failed: expected {expected:.4f}, got {loss.item():.4f}"
    )
    print("[OK] Test 2: clip activates when ratio > 1+epsilon (positive advantage)")

    # Test 3: When advantage is negative and ratio < 1-epsilon,
    # the clip should activate.
    log_probs_new = torch.tensor([-2.0])  # ratio = exp(-2) << 1-epsilon
    log_probs_old = torch.tensor([0.0])
    advantages = torch.tensor([-1.0])  # negative advantage
    loss = clipped_surrogate_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2)
    expected = -(1.0 - 0.2) * (-1.0)  # clipped at 1-epsilon
    assert abs(loss.item() - expected) < 1e-5, (
        f"Test 3 failed: expected {expected:.4f}, got {loss.item():.4f}"
    )
    print("[OK] Test 3: clip activates when ratio < 1-epsilon (negative advantage)")

    print()
    print("All tests passed. Open solutions/solution_01_clipped_loss.py to compare.")


if __name__ == "__main__":
    test_clipped_surrogate_loss()
