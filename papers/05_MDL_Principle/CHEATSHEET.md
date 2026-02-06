# MDL Principle Cheatsheet

Quick reference for Grünwald (2004) and our implementation.

---

## The Paper's Contribution (30 seconds)

**Problem:** How to select the best model from candidate models without overfitting?
**Insight:** Treat model selection as code selection. The best model compresses data the most.
**Result:** Three principled formulations (two-part, prequential, NML) that avoid arbitrary penalty terms.

This is an ~80-page tutorial/survey systematizing the MDL principle. The framework underlies Day 4's Bayesian weight compression and connects to Kolmogorov complexity.

---

## Key Equations

### Two-Part MDL (Crude)

```
Total cost = L(H) + L(D|H)
  L(H)   = k * b              (k params, b bits each)
  L(D|H) = (n/2) * log2(2*pi*e*var)   (Gaussian residuals)

Best model = argmin { L(H) + L(D|H) }
```

### Prequential MDL (Sequential)

```
L_preq = sum_i  -log2 P(x_i | x_1, ..., x_{i-1})
```

No model description needed. Cumulative prediction surprise.

### NML (Theoretical Gold Standard)

```
P_NML(x) = P(x | theta_hat(x)) / C_n

Stochastic complexity:
  COMP(M, n) ~ (k/2) * log(n / 2*pi)
```

---

## MDL vs AIC vs BIC

| Criterion | Penalty | Formula |
|-----------|---------|---------|
| AIC | 2k | -2*logL + 2k |
| BIC | k*log(n) | -2*logL + k*log(n) |
| MDL (NML) | ~(k/2)*log(n/2pi) | L(H) + L(D\|H) |

**Key difference:** AIC/BIC penalize each parameter equally. MDL adapts — a parameter near zero costs fewer bits.

---

## Quick Start

```bash
# Polynomial degree selection demo
python train_minimal.py

# Monte Carlo comparison (MDL vs AIC vs BIC)
python train_minimal.py --monte-carlo
```

### In Python

```python
from implementation import PolynomialMDL

# Fit a degree-3 polynomial and get its MDL score
model = PolynomialMDL(degree=3)
model.fit(X, y)
score = model.total_mdl(X, y)

# Or use the standalone function
from implementation import two_part_mdl_polynomial
total, model_cost, data_cost = two_part_mdl_polynomial(X, y, degree=3)
```

---

## Common Patterns

### Model Selection Loop

```python
best_score = float('inf')
for degree in range(1, max_degree + 1):
    coeffs = np.polyfit(X, y, degree)
    residuals = y - np.polyval(coeffs, X)
    score = (degree + 1) * 32 + (n/2) * np.log2(2*np.pi*np.e*np.var(residuals))
    if score < best_score:
        best_score, best_degree = score, degree
```

### Comparing Criteria

```python
def compare(log_likelihood, k, n):
    AIC = -2 * log_likelihood + 2 * k
    BIC = -2 * log_likelihood + k * np.log(n)
    MDL = -log_likelihood / np.log(2) + k * 32
    return AIC, BIC, MDL
```

---

## Debugging

### MDL selects too-simple model
- Check bits-per-parameter setting (32 is standard for float32)
- May need more data — MDL is conservative with small samples

### MDL selects too-complex model
- Residual variance may be underestimated
- Check that L(D|H) computation uses unbiased variance

### All criteria agree
- Normal for well-separated models. Disagreements show up when models are close in quality.

---

## What's From the Paper vs. Our Additions

| Concept | Source |
|---------|--------|
| Two-part codes: L(H) + L(D\|H) | Paper, Section 3 |
| Prequential MDL | Paper, Section 5 |
| NML / stochastic complexity | Paper, Section 6 |
| MDL vs AIC vs BIC comparison | Paper, Section 8 |
| Kolmogorov complexity connection | Paper, Section 9 |
| Polynomial degree selection example | Paper's running example |
| Monte Carlo robustness comparison | Our addition |
| All exercises | Our pedagogical additions implementing paper concepts |

---

*For paper details, see [paper_notes.md](paper_notes.md). For implementation guide, see [README.md](README.md).*
