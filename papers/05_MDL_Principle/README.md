# Day 5: A Tutorial Introduction to the Minimum Description Length Principle

> Grünwald (2004) — [arXiv:math/0406077](https://arxiv.org/abs/math/0406077)

**Time:** 3-4 hours
**Prerequisites:** Day 4 (Hinton's MDL for neural nets)
**Code:** NumPy + SciPy

---

## What This Paper Is Actually About

This is an ~80-page tutorial/survey by Peter Grünwald that systematizes the MDL principle — the idea that model selection is code selection, and the best model is the one that compresses data the most.

Day 4 (Hinton & van Camp) used MDL to regularize neural network weights. Today's paper explains the **mathematical foundation** underneath that idea: what "description length" means formally, how to compute it for different model classes, and why minimizing it selects good models.

### The paper covers:

1. **Two-part codes** — the simplest MDL formulation: L(H) + L(D|H)
2. **Prequential (predictive) codes** — MDL via sequential prediction
3. **Normalized Maximum Likelihood (NML)** — the minimax-optimal universal code
4. **Stochastic complexity** — NML codelength as a model complexity measure
5. **Connection to Kolmogorov complexity** — MDL as a practical approximation of algorithmic complexity
6. **Comparison with AIC, BIC, cross-validation**

---

## The Core Concept: Two-Part Codes

To communicate data $D$ to someone, you can either send it raw, or send a model plus residuals. If the model captures real structure, the total message is shorter.

**The MDL Principle:**

$$\text{Best model} = \arg\min_H \left[ L(H) + L(D \mid H) \right]$$

- $L(H)$ = bits to describe the model
- $L(D \mid H)$ = bits to describe the data given the model (residuals encoded as a code)

### Why this prevents overfitting

| Scenario | L(H) | L(D\|H) | Total |
|----------|-------|---------|-------|
| Too simple | Small | Large (poor fit) | Large |
| Just right | Moderate | Small (good fit) | **Minimum** |
| Too complex | Large (many params) | Near zero | Large |
| Extreme overfit | Exceeds raw data cost | Zero | Worse than no model |

No tuning parameter is needed — the sum automatically finds the balance.

### Encoding details

**Model description** (for parametric models):
$$L(H) = k \cdot b \text{ bits}$$
where $k$ = number of parameters, $b$ = bits per parameter.

**Data given model** (Gaussian residuals):
$$L(D \mid H) = \frac{n}{2}\log_2(2\pi e \hat{\sigma}^2)$$

---

## Prequential MDL

Instead of describing a model then encoding residuals, predict each point using the previous points:

$$L_{\text{preq}} = \sum_{i=1}^{n} -\log_2 P(x_i \mid x_1, \ldots, x_{i-1})$$

This is a valid code length (Kraft inequality holds). No model description is needed — the model class's complexity is implicitly penalized through prediction performance.

**Advantages over two-part codes:**
- No arbitrary model encoding scheme
- Natural for sequential/online data
- Connected to Bayesian model averaging via the plug-in code

---

## Normalized Maximum Likelihood (NML)

The paper's theoretical gold standard. Given model class $\mathcal{M}$:

$$P_{\text{NML}}(x^n) = \frac{P(x^n \mid \hat{\theta}(x^n))}{\sum_{y^n} P(y^n \mid \hat{\theta}(y^n))}$$

Numerator: best the model can do on this data. Denominator: sum over all possible datasets — measures model flexibility.

**Stochastic complexity** (the normalizing constant):
$$\text{COMP}(\mathcal{M}, n) \approx \frac{k}{2}\log\frac{n}{2\pi}$$

This resembles BIC's penalty but is derived from information theory rather than Bayesian approximation.

---

## MDL vs. AIC vs. BIC

The paper compares these in detail (Section 8).

| Criterion | Penalty per Parameter | Derivation | Key Property |
|-----------|-----------------------|-----------|-------------|
| AIC | 2 | KL divergence minimization | Efficient but overfits in small samples |
| BIC | $\log n$ | Laplace approx. to Bayesian marginal likelihood | Consistent (finds true model as $n \to \infty$) |
| MDL (NML) | $\approx \frac{1}{2}\log\frac{n}{2\pi}$ | Minimax optimal universal code | Consistent; adapts to parameter geometry |

**Key differences:**
- AIC's penalty doesn't grow with $n$ — it systematically overfits for large datasets
- BIC is close to MDL for regular exponential families
- MDL's penalty adapts to how parameters are used (a parameter near zero costs fewer bits)
- MDL doesn't assume a "true model" exists in the class

---

## Connection to Kolmogorov Complexity

MDL is the practical approximation of Kolmogorov complexity $K(x)$ — the length of the shortest program that produces $x$.

$K(x)$ is uncomputable (halting problem). MDL restricts to a computable model class and minimizes within it. As the model class grows, MDL codelengths approach $K(x)$.

Grünwald is careful to note: MDL captures statistical regularity relative to a model class. It does not claim to find "true understanding" in any philosophical sense. Domain knowledge matters for choosing the model class.

---

## The Architecture

```
MDL Model Selection Pipeline:

  Input: Data D, Candidate models {H_1, H_2, ..., H_k}

  For each model H_i:
    1. Compute L(H_i)     -- model description length
    2. Fit H_i to D       -- maximum likelihood
    3. Compute L(D|H_i)   -- residual code length
    4. Score = L(H_i) + L(D|H_i)

  Output: H* = argmin Score(H_i)
```

---

## What to Build

### Quick Start

```bash
python train_minimal.py                  # Polynomial degree selection demo
python train_minimal.py --monte-carlo    # Statistical comparison across trials
```

### Training Script

The demo generates noisy polynomial data (true degree = 3) and uses MDL, AIC, and BIC to select the degree. Expected output: MDL correctly picks degree 3.

---

## Exercises (in `Exercises/`)

| # | Task | What You'll Learn | Source |
|---|------|-------------------|--------|
| 1 | Two-Part Code | Basic MDL scoring for polynomials | Paper, Section 3 |
| 2 | Prequential MDL | Sequential prediction coding | Paper, Section 5 |
| 3 | Model Selection | Compare polynomial degrees with MDL | Paper, running example |
| 4 | NML Complexity | Compute stochastic complexity | Paper, Section 6 |
| 5 | MDL vs AIC vs BIC | Head-to-head comparison | Paper, Section 8 |

All exercises implement concepts directly from the paper. The Monte Carlo robustness test in exercise 5 is our extension.

Solutions in `solutions.py`.

---

## Key Takeaways

1. **MDL = compression-based model selection.** The best model minimizes total description length: model bits + residual bits.

2. **Three formulations, one principle.** Two-part codes (simple), prequential codes (sequential), NML (optimal) — all minimize description length.

3. **MDL adapts its penalty.** Unlike AIC (fixed 2k) or BIC (fixed k·log n), MDL's penalty depends on how parameters are used.

4. **Connection to Kolmogorov complexity.** MDL is the practical, computable approximation of algorithmic complexity.

5. **No "true model" assumption.** MDL works from compression alone — it doesn't assume the data was generated by any model in the class.

---

## Implementation Notes

Our code in `implementation.py` provides:

- `two_part_mdl_polynomial()` — the basic MDL score for regression (Section 3)
- `prequential_mdl_polynomial()` — sequential prediction coding (Section 5)
- `nml_complexity_polynomial()` — approximate stochastic complexity (Section 6)
- `PolynomialMDL` — model selection for polynomial regression (paper's running example)
- `compare_mdl_aic_bic()` — MDL vs AIC vs BIC scoring

The polynomial selection experiment is the paper's own running example throughout. The Monte Carlo comparison (running the experiment many times to compare accuracy across criteria) is our addition.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | MDL computation library: two-part, prequential, NML |
| `train_minimal.py` | Polynomial degree selection demo + Monte Carlo comparison |
| `visualization.py` | MDL score plots, fit comparisons, criterion comparison charts |
| `notebook.ipynb` | Interactive walkthrough |
| `Exercises/` | 5 exercises covering all three MDL variants |
| `paper_notes.md` | Detailed notes on Grünwald's tutorial |
| `CHEATSHEET.md` | Quick reference for MDL formulas |

---

## Further Reading

- [Grünwald (2004)](https://arxiv.org/abs/math/0406077) — this paper
- Grünwald, *The Minimum Description Length Principle* (MIT Press, 2007) — the full textbook
- Rissanen, *Stochastic Complexity in Statistical Inquiry* (1989) — the original MDL inventor
- [Hinton & van Camp (1993)](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) — Day 4, MDL applied to neural networks

---

**Next:** [Day 6](../06_Complexodynamics/)
