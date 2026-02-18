# Exercises: Proximal Policy Optimization

Five exercises covering the core components of PPO-Clip. Each exercise isolates one piece of the algorithm so you can understand it before assembling the full system.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Clipped Surrogate Objective | Easy (2/5) | 30 min |
| 2 | Generalized Advantage Estimation | Medium (3/5) | 45 min |
| 3 | Actor-Critic Network | Medium (3/5) | 45 min |
| 4 | Epsilon Ablation | Medium (3/5) | 60 min |
| 5 | PPO vs. Vanilla Policy Gradient | Hard (4/5) | 90 min |

## Exercise 1: Clipped Surrogate Objective

Implement the clipped surrogate loss from Equation 7 of the paper. Given a probability ratio and advantage estimates, compute the PPO-Clip objective. Understand when the clip activates and what it prevents.

## Exercise 2: Generalized Advantage Estimation

Implement GAE from scratch. Given a sequence of rewards, values, and dones, compute the advantage estimates using the backward recurrence. Understand how lambda controls the bias-variance tradeoff.

## Exercise 3: Actor-Critic Network

Build the actor-critic network from scratch. Implement both the actor (policy head) and critic (value head) with shared backbone. Verify that the network outputs valid probability distributions and value estimates.

## Exercise 4: Epsilon Ablation

Train PPO with different epsilon values (0.05, 0.1, 0.2, 0.3, 0.5) on CartPole-v1 and compare final performance. This reproduces the spirit of Table 1 in the paper.

## Exercise 5: PPO vs. Vanilla Policy Gradient

Remove the clipping from PPO (set epsilon=infinity, or equivalently use the unclipped objective) and compare training stability against full PPO-Clip. This demonstrates what the clipping actually buys you.

## How to Use

1. Read the exercise file — each has detailed instructions
2. Find the TODO sections — these are what you implement
3. Run the tests — each file has a test function at the bottom
4. Check solutions — compare with `solutions/solution_X.py`

## Tips

- Exercise 1 is the most important. The clip is the entire contribution of the paper.
- For Exercise 2, work through the math by hand on a 3-step sequence before coding.
- For Exercise 4, run each epsilon value for at least 50 iterations to see the difference.
- For Exercise 5, the instability of vanilla PG may not be obvious on CartPole (too easy). Try LunarLander-v2.

## Common Issues

- **Wrong sign in loss:** PPO maximizes the objective, but PyTorch minimizes. Make sure you negate the loss before calling .backward().
- **Old log probs not detached:** The old policy log probabilities must be detached from the computation graph. If they are not, the ratio gradient flows back through the old policy, which is wrong.
- **GAE backward loop:** The GAE recurrence runs backward through time (from t=T-1 to t=0). A common mistake is running it forward.
