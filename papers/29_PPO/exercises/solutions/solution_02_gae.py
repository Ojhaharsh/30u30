"""
Solution 2: Generalized Advantage Estimation (GAE)

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
    Compute GAE advantage estimates via backward recurrence.

    GAE formula (Schulman et al. 2015b):
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

    Computed via backward recurrence:
        gae = 0
        for t in reversed(range(T)):
            delta = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
            gae = delta + gamma * lambda * (1 - done_t) * gae
            A_t = gae

    The (1 - done_t) term is critical: it zeros out the bootstrap value
    at episode boundaries, so we don't propagate value estimates across
    episode resets.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
        # values[t+1] is the bootstrap value for the next state.
        # If done=1, the episode ended at step t, so there is no next state:
        # we zero out the bootstrap value.
        delta = (
            rewards[t]
            + gamma * values[t + 1] * (1.0 - dones[t])
            - values[t]
        )

        # GAE recurrence: accumulate discounted TD residuals.
        # The (1 - done_t) also zeros out the accumulated GAE at boundaries,
        # preventing advantage estimates from crossing episode boundaries.
        gae = delta + gamma * lambda_ * (1.0 - dones[t]) * gae
        advantages[t] = gae

    return advantages


# Run tests
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from exercises.exercise_02_gae import test_gae
    import exercises.exercise_02_gae as ex
    ex.compute_gae = compute_gae
    test_gae()
