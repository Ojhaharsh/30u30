# Day 30 Cheat Sheet: RLHF

## The Big Idea (30 seconds)
You train a reward model (RM) to predict what a human prefers, then use PPO to optimize the agent against that reward model.

---

## Quick Start
```bash
# 1. Run the full RLHF loop (using a synthetic oracle as the human)
python train_minimal.py --env CartPole-v1 --steps 10000

# 2. Visualize the learned reward vs. true reward
python visualization.py
```

---

## Key Hyperparameters (From Paper & InstructGPT)

| Parameter | Typical Value | What It Is |
|-----------|---------------|------------|
| Clip Length | 1-2 seconds | Length of video segments shown to humans. Too short = can't see behavior; too long = confusing. |
| Feedback Budget | ~1% of steps | How many labels you get. Christiano et al. used ~1,000 queries for Atari. |
| Reward LR | 1e-4 | Learning rate for the Reward Model. Usually smaller than PPO LR. |
| Batch Size | 64 pairs | Number of preference pairs per RM update. |
| Discount ($\gamma$) | 0.99 | Standard RL discount factor. |

---

## The Math (Copy-Paste Ready)

### Bradley-Terry Preference Model
$$ P[\sigma^1 \succ \sigma^2] = \frac{\exp(\sum r(s^1))}{\exp(\sum r(s^1)) + \exp(\sum r(s^2))} $$

### Preference Loss (Cross-Entropy)
$$ L = - E_{(\sigma^1, \sigma^2, \mu) \sim D} [\mu(1)\log P(1 \succ 2) + \mu(2)\log P(2 \succ 1)] $$

---

## Common Issues & Fixes

### 1. Reward Hacking / Hallucination
- **Symptom:** The agent learns a weird behavior that gets high reward from the RM but looks bad to you.
- **Fix:** Collect more feedback on these "weird" behaviors to teach the RM they are bad.

### 2. Mode Collapse
- **Symptom:** The agent outputs the exact same thing every time.
- **Fix:** Add KL-divergence penalty (from PPO) to keep the policy close to the initial policy (used in InstructGPT).

### 3. Reward Scale Unstable
- **Symptom:** PPO fails to converge.
- **Fix:** Normalize rewards to mean 0, std 1 (using a running mean/std).

---

## Debugging Checklist

- [ ] Is the Reward Model accuracy increasing? (Should go >60% quickly)
- [ ] Is the PPO agent improving on the *learned* reward?
- [ ] Is the *learned* reward correlated with the ground-truth reward? (Check scatter plot)
- [ ] Are preference labels consistent? (Simulated Oracle is perfect; humans are noisy)

---

## Experiment Ideas

1.  **Easy:** Run `train_minimal.py` and see if CartPole stays up using only preference feedback.
2.  **Medium:** Change the Oracle to prefer "moving left" instead of "staying up." See if the agent learns to move left.
3.  **Advanced:** Replace the Synthetic Oracle with a real CLI input where YOU press '1' or '2' to judge clips.
