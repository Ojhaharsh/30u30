# Day 27 Cheat Sheet: Machine Super Intelligence

## The Big Idea (30 seconds)

Intelligence is not about being "good at math" or "fast at Chess." It is the ability to achieve goals in a **wide range of environments**. To measure this universally, we sum an agent's performance across all possible computer programs, weighting them by their complexity ($2^{-K}$).

**Occam's Principle**: Simple environments are exponentially more likely and thus more important for defining general intelligence.

---

## Architecture: The Formal Loop

```
Environment Update: (o_t, r_t) ~ μ(h_{t-1}, a_t)
Agent Action:        a_{t+1}   ~ π(h_t)
Intelligence:        Υ(π)      = ∑_{μ ∈ E} 2^{-K(μ)} V_μ^π
```

**Key insight**: Intelligence is the **Average Performance** over the **Universal Distribution**.

---

## Quick Start (Project Specific)

### Benchmarking
```bash
# Run the core benchmark (Random vs RL vs Predictive)
python train_minimal.py

# Generate the Intelligence Spectrum visualization
python visualization.py
```

### In Python
```python
from implementation import UniversalIntelligenceMeasure, GridWorld, PredictiveAgent

# 1. Create a suite of environments
envs = [GridWorld(size=3), GridWorld(size=10)]

# 2. Initialize the Measure
measure = UniversalIntelligenceMeasure(envs)

# 3. Evaluate an Agent
results = measure.evaluate(PredictiveAgent())
print(f"Upsilon Score: {results['upsilon_normalized']}")
```

---

## Hyperparameter Guide (Simulation Proxies)

| Parameter | Typical Range | Description | Too Low | Too High |
|-----------|---------------|-------------|---------|----------|
| `complexity` ($K$) | 2 - 20 | Proxy for Kolmogorov complexity. | Too simple (trivial) | Too noisy (unlearnable) |
| `episodes` | 10 - 100 | Simulation length per environment. | High variance score | Slow evaluation |
| `lr` (RL Agent) | 0.01 - 0.1 | Learning rate for benchmarking agents. | Never learns pattern | Unstable / Oscillates |
| `history_len` | 5 - 50 | Memory window for Predictive agents. | Forgets patterns | CPU/Memory intensive |

---

## Common Issues & Fixes

### 1. Upsilon Score Underflows
**Symptom**: `upsilon_raw` becomes `0.0000000`
**Cause**: The $2^{-K}$ factor for high complexities is too small for float64.
**Fix**:
```python
# Evaluate in log-space or shift complexities
weight = 2**( - (K - K_min) )
```

### 2. Random Agent beats RL Agent
**Symptom**: Baseline outperforms your learner.
**Cause**: The environment suite is too complex ($K$ is high) and the learner hasn't had enough episodes to converge.
**Fix**: Increase `episodes` or use a more efficient agent like `PredictiveAgent`.

### 3. "Invariance" doesn't hold
**Symptom**: Ranking of agents changes when $K$ is shifted.
**Cause**: The environment suite $E$ is too small.
**Fix**: The Invariance Theorem is an *asymptotic* property. You need a larger $E$ to see the ranking stabilize.

---

## Debugging Checklist

- [ ] **Environments Normalized?** Rewards must be in $[0, 1]$.
- [ ] **Complexity Valid?** $K$ should reflect the "program size" of the env.
- [ ] **History Reset?** Ensure agents don't carry memory *between* different environments.
- [ ] **Occam Weighting Active?** Check if you are multiplying by $2^{-K}$.

---

## Visualization Guide

Use `visualization.py` to generate:
1. **The Spectrum**: A scatter plot of $V_{\mu}$ vs $K(\mu)$.
2. **Weight Bar Chart**: Showing the $2^{-K}$ distribution.
3. **Agent Ranking**: A bar chart of normalized $\Upsilon$ scores.

---

## When to use Universal Intelligence vs Other Metrics

| Metric | Origin | Best For | Weakness |
|--------|--------|----------|----------|
| **Upsilon** | Legg | AGI Researchers | Non-computable in limit |
| **Loss/Accuracy** | Standard DL | Specific Model training | Narrow (not general) |
| **Human Eval** | Chatbots | Chat/Social apps | Scale/Cost/Bias |
| **Gym Leaderboards** | OpenAI | Comparing RL algorithms | Overfitting to sets |

---

## Key Equations Summary

```
K(μ): Length of shortest program for μ
P(μ): 2^-K(μ) (Universal Prior)
V(π, μ): Expected total reward of π in μ
Υ(π): ∑ P(μ) V(π, μ)
```

---

## Resources
- **Thesis**: Legg (2008)
- **Code**: `implementation.py`
- **Day 26**: For the math of $K$.
- **Day 28**: For how $\Upsilon$ relates to Safety.

---

**Memory Reminder**: 
*"Intelligence is the average performance, but simple things matter most."* 
(This is the Day 27 equivalent of the "Cell state is the conveyor belt" mantra).
