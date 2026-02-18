# Day 29 Cheat Sheet: Proximal Policy Optimization

Quick reference for implementing and debugging PPO.

---

## The Big Idea (30 seconds)

PPO is a policy gradient algorithm that prevents catastrophic policy updates by clipping the probability ratio between the new and old policy. Instead of constraining the update (TRPO), it removes the incentive to update too far. The result is a stable, first-order algorithm that works across discrete and continuous action spaces.

---

## Quick Start

```bash
python setup.py                                    # verify environment
python implementation.py --demo                    # run CartPole demo
python train_minimal.py --env CartPole-v1 --epochs 100
python train_minimal.py --env LunarLander-v2 --epochs 300
```

---

## Key Hyperparameters

| Parameter | Paper Default | What It Does | Tips |
|-----------|--------------|--------------|------|
| `epsilon` | 0.2 | Clip range for probability ratio | 0.1-0.3 typical; smaller = more conservative |
| `gamma` | 0.99 | Discount factor | 0.99 for most tasks; 0.999 for long-horizon |
| `lambda` (GAE) | 0.95 | Bias-variance tradeoff in advantage | 0.9-0.99; lower = less variance, more bias |
| `lr` | 3e-4 | Adam learning rate | Anneal linearly to 0 for best results |
| `K_epochs` | 10 (MuJoCo), 3 (Atari) | Epochs per data batch | More epochs = more data reuse but instability risk |
| `T` | 2048 (MuJoCo), 128 (Atari) | Timesteps collected per iteration | Larger = more stable gradient estimates |
| `M` | 64 | Minibatch size | 32-256 typical |
| `c1` | 1.0 | Value function loss coefficient | Reduce if value loss dominates |
| `c2` | 0.01 (Atari), 0 (MuJoCo) | Entropy bonus coefficient | Higher = more exploration |

Source: Table 3 in the paper (MuJoCo) and Section 6.3 (Atari).

---

## The Clipped Objective (Equation 7)

```python
# r_t = ratio of new to old policy probabilities
r_t = torch.exp(log_probs_new - log_probs_old)

# clip the ratio
r_clipped = torch.clamp(r_t, 1 - epsilon, 1 + epsilon)

# take the pessimistic (minimum) of clipped and unclipped
policy_loss = -torch.min(r_t * advantages, r_clipped * advantages).mean()
```

The minus sign: we maximize the objective by minimizing its negative.

---

## GAE Advantage Estimation

```python
# compute TD residuals
deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]

# compute GAE advantages (backward pass)
advantages = []
gae = 0
for delta in reversed(deltas):
    gae = delta + gamma * lambda_ * gae
    advantages.insert(0, gae)

# normalize advantages within minibatch
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## Full Loss Function (Equation 9)

```python
# policy loss (clipped surrogate)
policy_loss = -torch.min(r_t * adv, r_clipped * adv).mean()

# value function loss
value_loss = F.mse_loss(values, returns)

# entropy bonus (encourages exploration)
entropy_bonus = dist.entropy().mean()

# combined loss
loss = policy_loss + c1 * value_loss - c2 * entropy_bonus
```

---

## Common Issues and Fixes

**Loss explodes / NaN gradients**
- Clip gradients: `torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)`
- Check that advantages are normalized
- Reduce learning rate

**Policy collapses to one action**
- Increase entropy coefficient `c2`
- Check that log probabilities are computed correctly (not probabilities)
- Verify that the ratio is computed as exp(log_new - log_old), not new/old directly

**Value function diverges**
- Reduce `c1` (value loss coefficient)
- Use separate learning rates for actor and critic
- Check that returns (targets for value function) are computed correctly

**Training is slow / not improving**
- Increase `T` (more data per iteration)
- Check that advantages are not all near zero (normalization issue)
- Verify that the environment rewards are not too sparse

**Ratio explodes (r_t >> 1 or << 0)**
- The old policy log probabilities must be detached from the computation graph
- Recompute log probabilities from the current policy at each epoch, using the stored actions

---

## The Math (Copy-Paste Ready)

```
Probability ratio:    r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

Clipped objective:    L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

GAE:                  A_t = sum_l (gamma*lambda)^l * delta_{t+l}
                      delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)

Full objective:       L = L_CLIP - c1 * L_VF + c2 * S[pi]
                      L_VF = (V(s_t) - V_target)^2
                      S[pi] = -sum_a pi(a|s) * log(pi(a|s))
```

---

## Debugging Checklist

- [ ] Advantages normalized within each minibatch (not globally)
- [ ] Old policy log probs detached from computation graph
- [ ] Log probs recomputed from current policy at each epoch
- [ ] Gradient clipping applied (max_norm=0.5)
- [ ] Observations normalized if using MuJoCo (running mean/std)
- [ ] Returns computed correctly (discounted sum of rewards, not just rewards)
- [ ] Value function targets detached from computation graph
- [ ] Entropy bonus has correct sign (subtract from loss, not add)

---

## Experiment Ideas

**Easy:**
- Train on CartPole-v1 (should solve in ~50 epochs)
- Plot the clipping frequency over training (how often does the clip activate?)

**Medium:**
- Ablate epsilon: train with epsilon in {0.05, 0.1, 0.2, 0.3, 0.5} and compare stability
- Compare PPO vs. vanilla policy gradient (remove the clip, keep everything else)

**Advanced:**
- Implement linear learning rate annealing and compare to constant lr
- Add observation normalization and test on LunarLander-v2
- Implement PPO-Penalty (adaptive KL) and compare to PPO-Clip

---

## File Reference

| File | Use It For |
|------|-----------|
| `implementation.py` | Core PPO: actor-critic, GAE, clipped loss |
| `train_minimal.py` | Training with CLI args |
| `visualization.py` | Learning curves, clip frequency, entropy |
| `exercises/` | 5 exercises with solutions |

---

## Success Criteria

- CartPole-v1: average reward > 475 over 100 episodes
- LunarLander-v2: average reward > 200 over 100 episodes
- Clipping frequency: should be non-zero but not too high (5-30% of steps)
- Entropy should decrease over training (policy becoming more deterministic)

---

**Previous:** [Day 28 — CS231n](../28_CS231n/)
**Next:** [Day 30 — RLHF](../30_RLHF/)
