# 📋 Day 5 Cheatsheet: MDL Principle

> Quick reference for Minimum Description Length model selection.

---

## 🎯 The One-Liner

**MDL:** Choose the model that gives the shortest total description of the data.

```
Best Model = argmin { L(Model) + L(Data | Model) }
```

---

## 📐 Core Formulas

### Two-Part Code (Basic MDL)

```python
MDL_score = L(H) + L(D|H)

# Where:
L(H)    = bits to describe the model (parameters, structure)
L(D|H)  = bits to describe data given model (residuals/likelihood)
```

### For Gaussian Residuals

```python
def two_part_mdl(n_params, residuals, bits_per_param=32):
    """
    Two-part MDL for regression with Gaussian errors.
    """
    n = len(residuals)
    variance = np.var(residuals)
    
    # Model description length
    L_H = n_params * bits_per_param
    
    # Data description length (Gaussian code)
    L_D_H = 0.5 * n * np.log2(2 * np.pi * np.e * variance)
    
    return L_H + L_D_H
```

### Prequential (Sequential) MDL

```python
def prequential_mdl(predictions, targets):
    """
    Predictive MDL: sum of prediction surprisals.
    """
    # For each point, how surprised were we?
    surprisals = -np.log2(predictions[targets])  # if classification
    # OR for regression with Gaussian:
    # surprisals = 0.5 * np.log2(2*pi*var) + (y - pred)^2 / (2*var*log(2))
    
    return np.sum(surprisals)
```

### NML Stochastic Complexity

```python
def nml_complexity(n_samples, n_params):
    """
    Approximate NML complexity for exponential family.
    
    COMP(n) ≈ (k/2) * log(n/2π) + log(Γ(k/2))
    """
    from scipy.special import gammaln
    k = n_params
    n = n_samples
    
    comp = 0.5 * k * np.log(n / (2 * np.pi))
    comp += gammaln(k / 2) / np.log(2)  # convert to bits
    
    return comp
```

---

## 🔄 The Three MDL Variants

| Variant | Formula | When to Use |
|---------|---------|-------------|
| **Two-Part** | L(H) + L(D\|H) | Simple model comparison |
| **Prequential** | Σ -log P(xᵢ\|x₁...xᵢ₋₁) | Online/sequential data |
| **NML** | -log P_NML(D) | Theoretical optimality |

---

## 📊 MDL vs AIC vs BIC

```python
def compare_criteria(log_likelihood, n_params, n_samples):
    """
    Compare model selection criteria.
    """
    k = n_params
    n = n_samples
    L = log_likelihood  # higher is better fit
    
    # Akaike Information Criterion
    AIC = -2 * L + 2 * k
    
    # Bayesian Information Criterion  
    BIC = -2 * L + k * np.log(n)
    
    # Two-Part MDL (approximate)
    MDL = -L / np.log(2) + k * 32  # 32 bits per param
    
    return {'AIC': AIC, 'BIC': BIC, 'MDL': MDL}
```

### Quick Comparison

| Criterion | Penalty per Parameter | Notes |
|-----------|----------------------|-------|
| AIC | 2 | Asymptotically efficient |
| BIC | log(n) | Consistent (finds true model) |
| MDL | Adaptive | Based on information theory |

---

## 🧮 Polynomial Regression Example

```python
import numpy as np

def polynomial_mdl(X, y, degree, bits_per_coef=32):
    """
    Compute MDL for polynomial regression.
    """
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    residuals = y - y_pred
    
    # Model complexity: (degree + 1) coefficients
    L_H = (degree + 1) * bits_per_coef
    
    # Data given model: Gaussian code for residuals
    n = len(y)
    variance = np.var(residuals) + 1e-10
    L_D_H = 0.5 * n * np.log2(2 * np.pi * np.e * variance)
    
    return L_H + L_D_H, {'L_H': L_H, 'L_D_H': L_D_H}

# Find best degree
for d in range(1, 11):
    score, details = polynomial_mdl(X, y, d)
    print(f"Degree {d}: {score:.0f} bits")
```

---

## 🎭 The Spy Code Analogy

```
Scenario: Transmit 365 temperature readings

Naive (no model):
  365 × 8 bits = 2920 bits

With average model (T = 15°C):
  Model: 10 bits (one number)
  Residuals: 365 × 6 bits = 2190 bits
  Total: 2200 bits ✓

With seasonal model (sine wave):
  Model: 50 bits (4 parameters)
  Residuals: 365 × 2 bits = 730 bits
  Total: 780 bits ✓✓

With overfit model (365-degree poly):
  Model: 3650 bits (365 coefficients)
  Residuals: 0 bits
  Total: 3650 bits ✗ (worse than naive!)
```

---

## 🔧 Quick Implementation Patterns

### Pattern 1: Model Selection Loop

```python
best_model = None
best_score = float('inf')

for model in candidate_models:
    model.fit(X, y)
    score = compute_mdl(model, X, y)
    
    if score < best_score:
        best_score = score
        best_model = model
```

### Pattern 2: Automatic Complexity Penalty

```python
def adaptive_penalty(model, X, y):
    """
    MDL-style adaptive penalty based on parameter precision.
    """
    # Parameters close to zero need fewer bits
    params = model.get_parameters()
    
    # Precision needed ∝ |param| / tolerance
    bits_needed = np.sum(np.log2(np.abs(params) / 0.01 + 1))
    
    return bits_needed
```

### Pattern 3: Prequential for Time Series

```python
def prequential_ar(data, order):
    """
    Prequential MDL for autoregressive model.
    """
    total_bits = 0
    
    for t in range(order, len(data)):
        # Fit on history
        X_hist = np.array([data[t-order:t]])
        model.partial_fit(X_hist, [data[t]])
        
        # Predict and score
        pred_prob = model.predict_proba(X_hist)
        total_bits += -np.log2(pred_prob[0, int(data[t])])
    
    return total_bits
```

---

## 📈 Visualization Quick Start

```python
import matplotlib.pyplot as plt

def plot_mdl_selection(degrees, mdl_scores, components):
    """
    Plot MDL model selection with breakdown.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    L_H = [c['L_H'] for c in components]
    L_D_H = [c['L_D_H'] for c in components]
    
    ax.bar(degrees, L_H, label='L(H) - Model', color='steelblue')
    ax.bar(degrees, L_D_H, bottom=L_H, label='L(D|H) - Data', color='coral')
    
    best_idx = np.argmin(mdl_scores)
    ax.axvline(degrees[best_idx], color='green', linestyle='--', 
               label=f'Best: degree {degrees[best_idx]}')
    
    ax.set_xlabel('Model Complexity (Polynomial Degree)')
    ax.set_ylabel('Description Length (bits)')
    ax.set_title('MDL Model Selection')
    ax.legend()
    
    plt.savefig('mdl_selection.png', dpi=150, bbox_inches='tight')
```

---

## ⚠️ Common Pitfalls

### ❌ Wrong: Fixed bits per parameter

```python
# Bad: Always 32 bits per parameter
L_H = n_params * 32
```

### ✅ Right: Adaptive precision

```python
# Better: Precision based on actual values
L_H = sum(bits_for_precision(p) for p in params)
```

### ❌ Wrong: Ignoring integer encoding

```python
# Bad: Treating discrete parameters as continuous
L_H = n_categories * 32
```

### ✅ Right: Universal integer code

```python
# Better: Use log*(n) for integers
def universal_integer_code(n):
    """Elias delta code length."""
    if n <= 0:
        return 1
    log_n = np.floor(np.log2(n)) + 1
    return log_n + 2 * np.floor(np.log2(log_n)) + 1
```

---

## 🧠 Key Intuitions

### Why L(H) + L(D|H)?

```
Total message = "Here's my theory" + "Here are the exceptions"
             = Model description + Residual encoding
```

### Why MDL avoids overfitting?

```
Overfit model:
  L(H) = HUGE (many parameters)
  L(D|H) = 0 (perfect fit)
  Total = HUGE ✗

Good model:
  L(H) = moderate (few parameters)
  L(D|H) = small (good fit)
  Total = small ✓
```

### Why compression = understanding?

```
Random data: Incompressible (no pattern to exploit)
Structured data: Compressible (pattern = model = understanding)

If you can compress it, you understand it.
```

---

## 🎯 Decision Flowchart

```
Start: "Which model should I use?"
  │
  ├─ Do you have candidate models?
  │   │
  │   ├─ Yes → Compute MDL for each, pick minimum
  │   │
  │   └─ No → Define model class, enumerate complexities
  │
  ├─ Is data sequential/online?
  │   │
  │   ├─ Yes → Use Prequential MDL
  │   │
  │   └─ No → Use Two-Part MDL
  │
  └─ Need theoretical guarantees?
      │
      ├─ Yes → Use NML (if computable)
      │
      └─ No → Two-Part is fine
```

---

## 📚 Quick Reference Table

| Concept | Formula | Intuition |
|---------|---------|-----------|
| Two-Part MDL | L(H) + L(D\|H) | Model + residuals |
| Prequential | Σ -log P(xᵢ\|past) | Cumulative surprise |
| NML | -log P_NML(D) | Minimax optimal |
| Stochastic Complexity | COMP(M, n) | "Price" of model class |
| Regret | MDL - best hindsight | How much worse than oracle |

---

## 🔗 Code Locations

| File | Purpose |
|------|---------|
| `implementation.py` | Core MDL functions |
| `visualization.py` | Plotting utilities |
| `train_minimal.py` | Quick demo |
| `Exercises/` | Hands-on practice |

---

*"Keep it simple. If your model is longer than the data, you're doing it wrong."*
