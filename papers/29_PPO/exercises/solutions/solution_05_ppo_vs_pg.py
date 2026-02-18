"""
Solution 5: PPO vs. Vanilla Policy Gradient

Reference: Schulman et al. (2017), Section 3
https://arxiv.org/abs/1707.06347
"""

import torch


def vanilla_pg_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    Vanilla policy gradient loss (REINFORCE with baseline).

    L_PG = -E[log_pi(a|s) * A_t]

    This is the baseline that PPO improves upon. The key difference from
    PPO: there is no probability ratio and no clipping. The gradient
    directly scales log_probs by the advantage, with no constraint on
    how much the policy can change.

    Note: log_probs_old is intentionally unused. In vanilla PG, we do
    not compute an importance sampling ratio — we just use the current
    policy's log probabilities directly. This is why vanilla PG can only
    do one gradient step per data batch (the data is on-policy only for
    the current policy, not for the updated policy).
    """
    # Vanilla PG: just scale log probs by advantages, no clipping.
    # The gradient of this loss w.r.t. theta is the policy gradient:
    # grad = E[grad_theta log_pi(a|s) * A_t]
    policy_loss = -(log_probs_new * advantages).mean()
    return policy_loss


# Expected findings when running exercise_05:
#
# PPO-Clip: stable training curve, reward increases monotonically.
# Vanilla PG: may reach high rewards but then collapse (reward drops suddenly).
#
# The collapse happens because vanilla PG can take arbitrarily large gradient
# steps. If the step is too large, the policy changes so much that the
# advantage estimates (computed from the old policy) are no longer valid,
# and the update is destructive.
#
# PPO's clipping prevents this by capping the probability ratio at [1-eps, 1+eps].
# Once the ratio hits the clip boundary, the gradient is zeroed out for that
# sample, preventing further movement in that direction.

if __name__ == "__main__":
    print("Solution 5: vanilla_pg_loss implemented.")
    print("Run exercise_05_ppo_vs_pg.py to see the comparison.")
    print()
    print("Key insight: vanilla PG loss = PPO loss with epsilon=infinity.")
    print("The clipping is the only difference between the two algorithms.")
