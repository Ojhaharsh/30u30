# Solutions: Day 29 (PPO)

This directory contains reference implementations for the 5 exercises.

## Overview

| Solution | Implements | Complexity | Key Insight |
|----------|------------|------------|-------------|
| `solution_01_clipped_loss.py` | Clipped Surrogate Objective | Easy | The min() logic is a pessimistic lower bound that prevents destructive updates. |
| `solution_02_gae.py` | Generalized Advantage Estimation | Medium | The backward recurrence allows efficient O(T) computation of multi-step advantages. |
| `solution_03_actor_critic.py` | Actor-Critic Network | Medium | Tanh activations and orthogonal initialization are standard for PPO on MuJoCo. |
| `solution_04_epsilon_ablation.py` | Epsilon Ablation Analysis | Medium | epsilon=0.2 is the "sweet spot" (Table 1 in paper); too small is slow, too large is unstable. |
| `solution_05_ppo_vs_pg.py` | Vanilla PG Baseline | Medium | Baseline comparison showing how vanilla PG collapses without the clipping mechanism. |

## How to Check Your Work

Run each solution file directly to verify against the included tests:

```bash
python solutions/solution_01_clipped_loss.py
python solutions/solution_02_gae.py
python solutions/solution_03_actor_critic.py
```

For exercises 4 and 5, run the exercise script itself (which contains the training loop) after reviewing the solution:

```bash
python exercise_04_epsilon_ablation.py
python exercise_05_ppo_vs_pg.py
```
