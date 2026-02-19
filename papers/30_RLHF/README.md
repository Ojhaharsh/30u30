# Day 30: Deep Reinforcement Learning from Human Feedback (RLHF)

> Christiano, Leike, Brown, Martic, Legg, Amodei (2017) — [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)

**Time:** 4-6 hours  
**Prerequisites:** Day 29 (PPO), basic understanding of preference learning  
**Code:** PyTorch

---

## What This Paper Is Actually About

This paper introduced the method that eventually enabled ChatGPT: training reinforcement learning agents using human feedback (preferences) instead of a hand-crafted reward function.

Before this paper, RL mostly solved tasks with clear scores (Atari points, robot distance). But for complex behaviors—like "do a backflip" or "write a helpful summary"—writing a math formula for the reward is nearly impossible.

The authors propose learning a **Reward Model** from human judgments. A human watches two video clips of the agent acting and picks the better one. The system learns a reward function that agrees with these human choices, then trains the agent to maximize that learned reward using PPO.

> **Note:** This paper aligns agents on Atari and MuJoCo tasks. The application to language models (InstructGPT) came later (Stiennon et al. 2020, Ouyang et al. 2022), but the core algorithm—learning a reward model from pairwise comparisons and optimizing with PPO—is identical.

---

## The Core Idea

The method has three steps (Section 2):

1.  **Collect Trajectories:** The agent interacts with the environment.
2.  **Human Feedback:** We show pairs of trajectory segments $(\sigma^1, \sigma^2)$ to a human. The human picks the preferred one.
3.  **Reward Modeling:** We train a neural network $\hat{r}_\psi$ to predict which segment the human prefers.
4.  **Optimization:** We use PPO to train the policy $\pi$ to maximize the predicted reward $\hat{r}_\psi$.

This cycle repeats. As the agent improves, the human provides feedback on new, more complex behaviors.

---

## What the Authors Actually Showed

1.  **Success on Invisible Objectives:** They trained an agent to do a backflip (Hopper) using only 900 bits of human feedback (Figure 4, Section 5.1). No reward function for "backflip" existed in the environment code.
2.  **Efficiency:** On Atari games, they achieved superhuman performance with < 1% of the feedback required by previous methods (Figure 1).
3.  **Robustness:** The learned reward function often shaped behavior better than the original "true" reward, as humans could encourage safer or more natural movement (Section 5.3).

---

## The Architecture

### 1. The Preference Model (Section 2.1)
The probability that a human prefers segment $\sigma^1$ over $\sigma^2$ is modeled by the Bradley-Terry model:

$$ \hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(s_t)}{\exp \sum \hat{r}(s_t) + \exp \sum \hat{r}(s_t')} $$

where $\hat{r}$ is the reward network.

### 2. The Reward Network
A standard neural network (CNN for Atari, MLP for MuJoCo) that maps a single state (or observation) to a scalar reward.

### 3. The Policy (Day 29)
The PPO agent (actor-critic) that optimizes the learned reward.

---

## Implementation Notes

-   **Synthetic Oracle:** In this implementation, we simulate the "human" using the environment's ground-truth reward. This allows us to train without pausing for thousands of manual inputs.
-   **Reward Normalization:** The learned reward scale is arbitrary. We normalize it to have zero mean and unit variance to keep the PPO value function stable (Section 2.2).
-   **Batching:** We collect preferences in large batches. The paper uses asynchronous collection; we adhere to a simpler synchronous loop (Collect -> Label -> Train RM -> Train Policy).

---

## What to Build

### Quick Start

```bash
# Verify setup
python setup.py

# Run the full RLHF loop (using synthetic human feedback)
python train_minimal.py --env CartPole-v1 --steps 10000
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | **Bradley-Terry Model** | Implement the probability function that drives ChatGPT's alignment. |
| 2 | **Preference Loss** | Implement the cross-entropy loss for pairwise comparisons. |
| 3 | **Reward Normalization** | Build the running mean/std tracker to stabilize training. |
| 4 | **Synthetic Oracle** | Create the simulated human that provides labels. |
| 5 | **RLHF Loop** | Stitch the Reward Model and PPO agent together. |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1.  **Preferences satisfy the Bradley-Terry model.** We assume the probability of preference is proportional to the exponentiated sum of rewards. (Section 2.1)
2.  **Trajectory segments are the unit of feedback.** We don't grade single frames; we grade short clips (1-2 seconds). This captures dynamics (speed, jump height). (Section 3)
3.  **The Reward Model is non-stationary.** As the policy shifts, we need new feedback on the new behavior. (Section 2.2)
4.  **This is the "Alignment" recipe.** 99% of modern LLM alignment (RLHF) follows this exact pattern: Collect data -> Train Reward Model -> PPO.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Complete RLHF pipeline (Reward Model, Trainer, PPO wrapper) |
| `train_minimal.py` | Training script for CartPole/LunarLander |
| `visualization.py` | Plotting reward correlation and accuracy |
| `notebook.ipynb` | Interactive walkthrough of the preference logic |
| `exercises/` | 5 step-by-step exercises |

---

## Further Reading

-   [Original Paper](https://arxiv.org/abs/1706.03741) - Christiano et al. (2017)
-   [Learning to Summarize](https://arxiv.org/abs/2009.01325) - Stiennon et al. (2020) (Applied this to text)
-   [InstructGPT](https://arxiv.org/abs/2203.02155) - Ouyang et al. (2022) (The paper that made it famous)

---

**This is the final day of 30u30. You made it!**
