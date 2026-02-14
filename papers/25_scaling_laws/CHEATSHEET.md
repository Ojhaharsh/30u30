# Scaling Laws for NLM: Cheat Sheet

Quick reference for the fundamental constants, equations, and engineering heuristics from Kaplan et al. (2020).

---

## The Big Idea (30 seconds)

Performance ($L$) improves smoothly as a **power-law** with scale ($N, D, C$), provided you aren't bottlenecked by any single factor. These laws allow for precise performance prediction of massive models from small-scale experiments. The "Physics" of AI means that if you know how 1M parameters behave, you can predict how 1B parameters will behave.

---

## Architecture: The 12Ld² Rule

Kaplan et al. calculate the model size $N$ (excluding embeddings) using this heuristic:

```text
N ≈ 12 * L * d_model^2
```

### Where does the 12 come from?
For each of the $L$ layers:
- **Attention (4d²)**: $W_q, W_k, W_v, W_o$ are each $d \times d$.
- **Feed-Forward (8d²)**: First linear is $d \times 4d$, second is $4d \times d$.
- **Total per layer**: $4d^2 + 8d^2 = 12d^2$.

---

## Quick Start

### 1. Training a Sweep
```bash
# Run the scaling sweep simulation
python train_minimal.py

# Custom sweep with specific model sizes
python train_minimal.py --d_models 64 128 256 512
```

### 2. Visualization
```bash
# Generate the 4-panel Scaling Dashboard
python visualization.py
```

### 3. In Python
```python
from implementation import MasterFitter

ns = [1e6, 1e7, 1e8]
ls = [3.5, 3.1, 2.8]

fitter = MasterFitter(ns, ls)
fitter.fit()
print(f"Power-law exponent (alpha): {fitter.alpha}")
```

---

## Hyperparameter Guide

| Parameter | Symbol | OpenAI Value (WebText2) | Note |
|---|---|---|---|
| Model Parameters | $N$ | $\alpha_N = 0.076$ | Excludes embeddings |
| Dataset Size | $D$ | $\alpha_D = 0.095$ | Tokens seen |
| Training Compute | $C$ | $\alpha_C = 0.050$ | Measured in PF-days |
| Irreducible Loss | $L_\infty$ | $1.69$ | Limits of language entropy |

### Good Starting Point for Sweeps
- **n_layers**: Keep constant (e.g., 2-4) to see pure $d_{model}$ scaling.
- **n_heads**: 4-8 (doesn't significantly impact scaling).
- **d_model**: Sweep in powers of 2 (64, 128, 256, 512).

---

## Common Issues & Fixes

### 1. "The Overfitting Knee"
**Symptom**: Your scaling curve starts straight but suddenly "kinks" and flattens or goes up.
**Cause**: You have run out of unique data ($D$-bottleneck). You are training on the same data multiple times and the model is memorizing.
**Fix**: Increase your dataset size $D$ or use a smaller model $N$ to find the straight-line region.

### 2. "Wrong Slope"
**Symptom**: Your fitted $\alpha$ is much higher or lower than 0.076.
**Cause**: Small models on toy tasks often have much steeper scaling laws than massive models on natural language.
**Fix**: Use more complex training data (Natural Language) or ensure your Cross Entropy base is $e$ (natural log).

### 3. "Compute Mismatch"
**Symptom**: Your $6N$ calculation doesn't match your training logs.
**Cause**: Make sure you are using **Non-Embedding** parameters for $N$.
**Fix**: `n_params = total_params - (vocab_size * d_model)`.

---

## Debugging Checklist

- [ ] **Embedding Exclusion**: Are you sure you subtracted $V \times d$ from $N$?
- [ ] **Log-Log Scaling**: Are both axes logarithmic when you check for linearity?
- [ ] **Precision Check**: Is your transformer implementation matching the $12Ld^2$ heuristic within 1%?
- [ ] **PF-Days Scale**: $1 \text{ PF-day} = 10^{15} \times 24 \times 3600 \text{ FLOPS}$.
- [ ] **Irreducible Loss**: Did you include $L_\infty$ in your fitter? (Pure power laws lead to $L \to 0$, which is impossible).

---

## When to Use Scaling Laws

| Model Stage | Best For | Why? |
|---|---|---|
| **Research** | Tiny models (100k-1M) | Find the law, predict the future. |
| **Budgeting** | Calculating Cloud Costs | Know exactly how many GPU-days you need. |
| **Optimization** | Performance Frontiers | Decide if you should buy more data or more GPUs. |

---

## Key Equations Summary

| Concept | Equation | Purpose |
|---|---|---|
| **Compute** | $C \approx 6N \times B \times S \times T$ | Budgeting training runs. |
| **N-Scaling** | $L(N) = (N_c / N)^\alpha$ | Predicting gain from bigger models. |
| **Master Law** | $L(N, D) = [ (N_c/N)^{\frac{\alpha_N}{\alpha_D}} + D_c/D ]^{\alpha_D}$ | The "Universal" scaling equation. |
| **12Ld²** | $N \approx 12 \cdot L \cdot d_{model}^2$ | Heuristic for parameter count. |

---

## Resources

- **Original Paper**: [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361)
- **DeepMind Refinement**: [Hoffmann et al. (2022)](../26_chinchilla/)
- **Implementation**: `implementation.py`
- **Dashboard**: `visualization.py`

---

**Next Up:** [Chinchilla Scaling Laws](../26_chinchilla/)
