"""
Solution 1: Clipped Surrogate Objective

Reference: Schulman et al. (2017), Equation 7
https://arxiv.org/abs/1707.06347
"""

import torch


def clipped_surrogate_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Clipped surrogate objective (Equation 7 from the paper).

    L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

    The ratio r_t = pi_new / pi_old is computed in log space for
    numerical stability: r_t = exp(log_new - log_old).
    """
    # Probability ratio: r_t = pi_theta(a|s) / pi_theta_old(a|s)
    # Computed in log space to avoid numerical issues with small probabilities.
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Unclipped surrogate: r_t * A_t
    surr1 = ratio * advantages

    # Clipped surrogate: clip(r_t, 1-eps, 1+eps) * A_t
    # The clip prevents the ratio from going too far from 1.
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # Pessimistic lower bound: take the minimum of clipped and unclipped.
    # This means: when advantage is positive, we cap the gain at (1+eps)*A.
    # When advantage is negative, we cap the gain at (1-eps)*A.
    # The agent gets no benefit from pushing the ratio further.
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


# Run tests
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from exercises.exercise_01_clipped_loss import test_clipped_surrogate_loss

    # Monkey-patch the exercise module to use our solution
    import exercises.exercise_01_clipped_loss as ex
    ex.clipped_surrogate_loss = clipped_surrogate_loss
    test_clipped_surrogate_loss()
