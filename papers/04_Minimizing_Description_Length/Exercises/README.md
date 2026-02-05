# Exercises: MDL / Bayesian Neural Networks

5 hands-on exercises building on Hinton & van Camp (1993). All exercises are our pedagogical additions — the paper itself is theoretical with a minimal experiment. Exercises 3 and 5 are most directly connected to the paper's core ideas (the complexity-accuracy trade-off and the MDL principle).

**Time estimates:**
- Medium (20-30 min)
- Hard (30-40 min)

---

## Exercise 1: The Reparameterization Trick [Medium]

**Our addition** — the reparameterization trick is from Kingma & Welling (2014), not the 1993 paper. We use it because it's how you actually train Bayesian NNs with gradient descent.

**File:** `exercise_01_reparameterization.py`

**What you'll learn:**
- How to sample weights differentiably: $w = \mu + \sigma \cdot \epsilon$
- Why softplus ensures $\sigma > 0$
- How to verify sampling with statistical tests

**Tasks:**
1. Implement `softplus(x)` = $\log(1 + e^x)$
2. Implement `sample_gaussian(mu, rho, n_samples)` using the reparameterization trick
3. Verify: sampled mean should be close to $\mu$, sampled std close to $\sigma$
4. Check: when $\rho = 0$, $\sigma \approx 0.693$ (= $\log 2$)

**Success criteria:**
- Statistical tests pass (mean within 0.05 of $\mu$, std within 0.05 of $\sigma$)
- Different $\rho$ values produce different uncertainty levels

---

## Exercise 2: The Gap Experiment [Hard]

**Our addition** — demonstrates epistemic uncertainty, which is the practical payoff of the paper's ideas.

**File:** `exercise_02_gap_experiment.py`

**What you'll learn:**
- Training on data with missing regions reveals what the network doesn't know
- Monte Carlo forward passes produce uncertainty estimates
- Epistemic uncertainty (model doesn't know) vs. aleatoric uncertainty (data is noisy)

**Tasks:**
1. Generate sine wave data with a gap in the middle ($|x| < 1$)
2. Train the Bayesian network on gappy data
3. Run 100 forward passes on full range including the gap
4. Plot mean prediction with uncertainty band

**What to look for:**
- Tight prediction band where data exists
- Wide uncertainty band in the gap
- A standard NN would be confidently wrong in the gap; this shows honest uncertainty

---

## Exercise 3: Beta Parameter Study [Medium]

**Our addition** — but directly explores the paper's core equation ($\mathcal{L} = \text{error} + \beta \cdot KL$).

**File:** `exercise_03_beta_parameter.py`

**What you'll learn:**
- $\beta$ (kl_weight) controls the complexity-accuracy trade-off from the paper
- Low $\beta$: overfits (confident but wrong outside training data)
- High $\beta$: underfits (uncertain but safe)
- How to measure calibration

**Tasks:**
1. Train models with different $\beta$ values (0.001, 0.01, 0.1, 0.5, 1.0)
2. Measure test MSE and prediction uncertainty for each
3. Find the "sweet spot" where both are reasonable

**Tuning guide:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model predicts flat line | $\beta$ too high | Reduce $\beta$ |
| No uncertainty in gaps | $\beta$ too low | Increase $\beta$ |
| Wiggly, overfitting | $\beta$ too low | Increase $\beta$ |

---

## Exercise 4: Monte Carlo Predictions [Medium]

**Our addition** — standard practice for using Bayesian NNs, not discussed in the 1993 paper.

**File:** `exercise_04_monte_carlo.py`

**What you'll learn:**
- In production, you run the network many times and aggregate
- Uncertainty estimates converge with more samples
- The Law of Large Numbers in action

**Tasks:**
1. Run the network N times with different weight samples
2. Track how mean and std change as N increases (1, 5, 10, 50, 100)
3. Visualize convergence

**What to look for:**
- 1 sample: noisy, jagged
- 10 samples: clearer but still wobbly
- 100 samples: smooth and stable
- Rule of thumb: 50-100 samples is usually enough

---

## Exercise 5: Pareto Frontier [Hard]

**Our addition** — but directly visualizes the MDL principle from the paper.

**File:** `exercise_05_advanced_mdl.py`

**What you'll learn:**
- MDL loss = reconstruction loss + KL divergence
- The Pareto frontier shows you can't improve both simultaneously
- Compression ratio changes with $\beta$

**Tasks:**
1. Train models across a range of $\beta$ values
2. For each, record reconstruction error (MSE) and complexity cost (KL)
3. Plot the Pareto frontier: x-axis = KL (complexity), y-axis = MSE (error)
4. Find the "elbow" — often the best practical choice

**What to look for:**
- The frontier is a smooth curve from upper-left (simple, high error) to lower-right (complex, low error)
- Points below the frontier are Pareto-optimal
- The elbow represents the paper's insight: maximum compression with acceptable error

---

## Summary

| # | Topic | File | Time | Difficulty | Paper Connection |
|---|-------|------|------|------------|------------------|
| 1 | Reparameterization | `exercise_01_*.py` | 20-30 min | Medium | Kingma 2014 technique |
| 2 | Gap experiment | `exercise_02_*.py` | 30-40 min | Hard | Epistemic uncertainty |
| 3 | Beta parameter | `exercise_03_*.py` | 20-30 min | Medium | Core equation from paper |
| 4 | MC predictions | `exercise_04_*.py` | 20-30 min | Medium | Modern practice |
| 5 | Pareto frontier | `exercise_05_*.py` | 30-40 min | Hard | MDL principle visualization |

**Total time:** 2-3 hours

---

## Running

```bash
cd Exercises
python exercise_01_reparameterization.py
python exercise_02_gap_experiment.py
# etc.
```

Solutions are in `solutions.py` and `solutions_extra.py`. Try the exercises first.

---

## Related Files

- `solutions.py` — Reference solutions with explanations
- `solutions_extra.py` — Additional solution implementations
- `../implementation.py` — Full Bayesian NN implementation
- `../paper_notes.md` — Detailed paper notes
- `../CHEATSHEET.md` — Quick reference
