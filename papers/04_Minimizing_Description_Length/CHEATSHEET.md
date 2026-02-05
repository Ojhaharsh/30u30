# MDL / Bayesian NN Cheatsheet

Quick reference for Hinton & van Camp (1993) and our implementation.

---

## The Paper's Contribution (30 seconds)

**Problem:** Standard NNs use high-precision weights that overfit by encoding training noise.
**Insight:** Represent weights as Gaussian distributions. Minimize total description length = data error + weight complexity.
**Result:** MDL objective = variational free energy. The "bits back" argument connects information theory to Bayesian inference.

This is a theoretical paper (6 pages, one toy experiment). The framework it establishes is the foundation for VAEs, Bayes by Backprop, and much of Bayesian deep learning.

---

## Key Equations

```
Weight sampling:     w = mu + sigma * epsilon,    epsilon ~ N(0,1)
Sigma from rho:      sigma = log(1 + exp(rho))    (softplus, ensures sigma > 0)

KL divergence:       KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
Error cost:          NLL ~ MSE (assuming Gaussian noise)

Total loss:          L = NLL + beta * KL
```

- **KL** pulls weights toward prior N(0,1) — makes them simpler/fuzzier
- **NLL** pulls weights toward fitting data — makes them precise
- **beta** controls the trade-off (most important hyperparameter)

---

## Quick Start

```bash
# Train with default settings (gappy sine wave)
python train_minimal.py

# More regularization (simpler model, more uncertainty)
python train_minimal.py --kl-weight 0.5

# Less regularization (fits data harder, less uncertainty)
python train_minimal.py --kl-weight 0.001
```

### In Python

```python
from implementation import MDLNetwork

net = MDLNetwork(input_size=1, hidden_size=20, output_size=1)

# Forward pass — returns DIFFERENT result each time (weights are sampled)
pred = net.forward(x)

# Loss = error + beta * complexity
loss = MSE(pred, y) + kl_weight * net.total_kl()

# Prediction with uncertainty (100 forward passes)
mean, std = net.predict_with_uncertainty(x_test, n_samples=100)
```

---

## Paper's Hyperparameters

The paper is theoretical and doesn't specify practical training details. These are **our** recommended settings:

### Starting Point
```python
hidden_size = 20
kl_weight = 0.1       # beta — adjust first if problems
lr = 0.01
epochs = 2000
rho_init = -3.0       # sigma starts at ~0.05
```

---

## Hyperparameter Guide

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| `kl_weight` (beta) | 0.001 - 1.0 | Most important. Higher = simpler, fuzzier. Lower = overfits. |
| `hidden_size` | 10 - 50 | Network capacity |
| `rho_init` | -3.0 to -5.0 | Initial uncertainty. softplus(-3) ~ 0.05 |
| `lr` | 0.001 - 0.01 | Learning rate |

---

## Common Issues

### Model predicts a flat line (underfitting)
- `kl_weight` too high — the model is terrified of complexity
- Fix: reduce `kl_weight` (try 0.1 -> 0.01)

### No uncertainty in gaps (overfitting)
- `kl_weight` too low — acting like a standard NN
- Fix: increase `kl_weight`

### Loss explodes (NaN)
- sigma blew up or collapsed
- Fix: clip gradients, lower learning rate, check softplus for numerical issues

---

## Standard vs. Bayesian Comparison

| Aspect | Standard NN | MDL / Bayesian NN |
|--------|------------|-------------------|
| Weights | Fixed numbers | Distributions N(mu, sigma) |
| Prediction | Deterministic | Stochastic (different each pass) |
| Unknown data | Confidently wrong | Shows high uncertainty |
| Parameters | W, b | mu, rho (2x params per weight) |
| Loss | Error only | Error + KL divergence |

---

## Debugging Checklist

- [ ] Is beta (kl_weight) reasonable? Not too high (flat line) or too low (overfit)?
- [ ] Is sigma computed via softplus (always positive)?
- [ ] Are gradients combining BOTH error and KL sources?
- [ ] Is KL scaled by 1/N (number of data points)?
- [ ] For uncertainty plots: using enough MC samples (50-100)?

---

## What's From the Paper vs. Our Additions

| Concept | Source |
|---------|--------|
| Gaussian weight distributions | Paper |
| KL divergence as complexity cost | Paper |
| Bits back argument | Paper |
| MDL = variational free energy | Paper |
| Reparameterization trick | Kingma & Welling (2014) |
| Softplus parameterization | Modern practice |
| Monte Carlo uncertainty | Modern practice |
| Gap experiment | Our addition |
| All exercises | Our additions |

---

*For paper details, see [paper_notes.md](paper_notes.md). For implementation guide, see [README.md](README.md).*
