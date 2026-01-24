# Day 5: A Tutorial Introduction to the Minimum Description Length Principle

> *"The best hypothesis for a given data set is the one that leads to the best compression of the data."*
> — Peter Grünwald

**Paper:** [A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077)  
**Author:** Peter Grünwald (2004)  
**Core Idea:** Model selection is code selection. The best model is the one that compresses data the most.

**⏱️ Time to Complete:** 3-4 hours

**🎯 What You'll Learn:**
- Why compression = understanding (the deep insight)
- Two-Part Codes: L(H) + L(D|H) framework
- Prequential MDL for sequential prediction
- NML complexity (the theoretical gold standard)
- How MDL compares to AIC and BIC
- The connection to Kolmogorov Complexity
- Why Occam's Razor is mathematically optimal

---

## 🎯 The Big Idea (In Plain English)

Yesterday (Day 4), we saw that Hinton used MDL to regularize neural networks. Today, we dive into the **mathematical foundation** of MDL itself.

**The Central Question:**
> Given multiple hypotheses (models) that could explain my data, how do I choose the best one?

**The MDL Answer:**
> Convert each hypothesis into a *code* (compression scheme). The hypothesis whose code compresses the data the most is the best.

**Why this matters:**
- **Occam's Razor, formalized**: Simpler explanations are literally shorter codes
- **Overfitting, solved**: Complex models have long descriptions, penalized automatically
- **No arbitrary parameters**: Unlike AIC/BIC, MDL emerges from first principles
- **Universal**: Works for ANY model class (polynomials, neural networks, HMMs)

---

## 🧠 The Core Concept: "Two-Part Codes"

Imagine you're a spy sending a secret message. You have two choices:

### Strategy 1: Raw Transmission
Send every data point exactly as-is.
- **Cost**: Proportional to data size
- **No compression**: Wasteful if there's pattern

### Strategy 2: Two-Part Code
1. First, send a **description of the pattern** (the model)
2. Then, send the **deviations from the pattern** (the residuals)

```
Total Message = Model Description + Data Given Model
     L(H)      =      L(H)        +     L(D|H)
```

**The MDL Principle:**
> Choose the hypothesis H that minimizes L(H) + L(D|H)

---

## 🎭 The Spy Analogy (A Complete Story)

You're a spy who must transmit temperature readings from Moscow to HQ.

### Scenario: 365 daily temperatures

**Naive Approach (No Model):**
```
Send: 23.1, 24.5, 22.3, 25.1, ... (365 numbers)
Cost: 365 × 8 bits = 2920 bits
```

**Approach 1: Simple Model (Average)**
```
Model: "Temperature is always 15°C"
Cost of model: ~10 bits (just one number)
Cost of residuals: 365 × 6 bits = 2190 bits (deviations from 15)
Total: 2200 bits ✓ Better!
```

**Approach 2: Seasonal Model (Sine Wave)**
```
Model: "T(day) = 15 + 20*sin(2π*day/365)"
Cost of model: ~50 bits (amplitude, period, phase, offset)
Cost of residuals: 365 × 2 bits = 730 bits (small deviations)
Total: 780 bits ✓✓ Much better!
```

**Approach 3: Overfit Model (365-degree polynomial)**
```
Model: Polynomial that passes through every point exactly
Cost of model: 365 × 10 bits = 3650 bits (365 coefficients!)
Cost of residuals: 0 bits (perfect fit)
Total: 3650 bits ✗ Worse than naive!
```

**The Lesson:**
- Model 2 wins because it captures the TRUE pattern (seasons)
- Model 3 overfits: perfect fit, but model is longer than the data itself!
- MDL automatically finds the sweet spot

---

## 📐 The Mathematical Framework

### The Three Flavors of MDL

| Variant | Idea | Use Case |
|---------|------|----------|
| **Two-Part Code** | L(H) + L(D\|H) | Simple, intuitive |
| **Prequential Code** | Predict each point using previous | Online learning |
| **Normalized Maximum Likelihood (NML)** | Minimax optimal | Theoretical gold standard |

### 1. Two-Part Code (Crude MDL)

The simplest version:
```
MDL(H) = L(H) + L(D|H)
       = log₂(|Model Space|) + Σ log₂(1/P(xᵢ|H))
```

**Example: Polynomial Regression**
- L(H): Number of bits to specify polynomial degree + coefficients
- L(D|H): Sum of squared errors encoded as bits

```python
def two_part_mdl(model, data, residuals):
    """
    Compute two-part MDL score.
    
    L(H): bits to describe model
    L(D|H): bits to describe data given model
    """
    # Model complexity (number of parameters × precision)
    n_params = len(model.parameters())
    bits_per_param = 32  # float32
    L_H = n_params * bits_per_param
    
    # Data given model (residuals encoded as Gaussian)
    variance = np.var(residuals)
    n = len(residuals)
    L_D_given_H = 0.5 * n * np.log2(2 * np.pi * variance) + \
                  0.5 * np.sum(residuals**2) / (variance * np.log(2))
    
    return L_H + L_D_given_H
```

### 2. Prequential Code (Sequential MDL)

Instead of fitting a model then encoding, **predict each point using previous points**:

```
L_prequential = Σᵢ log₂(1/P(xᵢ | x₁, ..., xᵢ₋₁))
```

**Why this is powerful:**
- No arbitrary model description length
- Works for any probabilistic model
- Natural for time series

```python
def prequential_mdl(model_class, data):
    """
    Compute prequential (predictive) MDL.
    
    Train on x₁...xₙ₋₁, predict xₙ, accumulate log loss.
    """
    total_bits = 0
    
    for t in range(1, len(data)):
        # Train on data up to t-1
        model = model_class()
        model.fit(data[:t])
        
        # Predict point t
        prob = model.predict_proba(data[t])
        total_bits += -np.log2(prob)
    
    return total_bits
```

### 3. Normalized Maximum Likelihood (NML)

The **theoretically optimal** universal code:

```
P_NML(x) = P(x | θ̂(x)) / Σ_y P(y | θ̂(y))
```

Where θ̂(x) is the ML estimate for data x.

**Intuition:** Weight each model by how well it *could have* fit the data.

```python
def nml_complexity(model_class, n_samples, n_features):
    """
    Compute NML stochastic complexity (the normalizing constant).
    
    This is the "price" of using model_class for n samples.
    """
    # For Gaussian with unknown mean and variance:
    # COMP(n) ≈ (k/2) * log(n/2π) + log(Γ(k/2))
    k = n_features  # number of parameters
    complexity = 0.5 * k * np.log(n_samples / (2 * np.pi))
    complexity += np.log(gamma(k/2))
    return complexity
```

---

## 🔬 MDL vs. Other Model Selection Criteria

### The Big Three Comparison

| Criterion | Formula | Penalty | Origin |
|-----------|---------|---------|--------|
| **AIC** | -2·log(L) + 2k | 2k | Information theory (asymptotic) |
| **BIC** | -2·log(L) + k·log(n) | k·log(n) | Bayesian approximation |
| **MDL** | L(H) + L(D\|H) | Adaptive | Kolmogorov complexity |

### Why MDL is Different

**AIC/BIC:** Fixed penalty per parameter  
**MDL:** Penalty depends on how parameters are *used*

**Example: The "Irrelevant Feature" Problem**

You're fitting: y = β₁x₁ + β₂x₂ + noise

But x₂ is irrelevant (β₂ ≈ 0).

| Criterion | Behavior |
|-----------|----------|
| AIC | Penalizes β₂ by fixed 2 |
| BIC | Penalizes β₂ by fixed log(n) |
| MDL | β₂ ≈ 0 means it costs ~0 bits (adaptive!) |

**MDL recognizes that specifying "β₂ = 0" is cheap!**

---

## 🏗️ Architecture: The MDL Model Selector

```
┌─────────────────────────────────────────────────────────────┐
│                    MDL Model Selection                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Data D, Candidate Models {H₁, H₂, ..., Hₖ}          │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  For each model Hᵢ:                                  │    │
│  │                                                      │    │
│  │  1. Compute L(Hᵢ)     [Model Description Length]    │    │
│  │     └── # parameters × bits per parameter           │    │
│  │     └── + structure description                     │    │
│  │                                                      │    │
│  │  2. Fit Hᵢ to D       [Maximum Likelihood]          │    │
│  │     └── θ* = argmax P(D|θ, Hᵢ)                      │    │
│  │                                                      │    │
│  │  3. Compute L(D|Hᵢ)   [Data Description Length]     │    │
│  │     └── -log P(D|θ*, Hᵢ)                            │    │
│  │     └── = Negative Log Likelihood                   │    │
│  │                                                      │    │
│  │  4. Score(Hᵢ) = L(Hᵢ) + L(D|Hᵢ)                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Output: H* = argmin Score(Hᵢ)                              │
│          (Model with shortest total description)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Key Insight: Kolmogorov Complexity Connection

MDL is the **practical approximation** of Kolmogorov Complexity.

**Kolmogorov Complexity K(x):**
> The length of the shortest program that outputs x

**Problem:** K(x) is uncomputable! (Halting problem)

**MDL Solution:**
> Restrict to a *computable* model class, then minimize description length within that class.

```
K(x) ≈ min over H { L(H) + L(x|H) }
```

**The philosophical point:**
- K(x) is the "true" complexity of x
- MDL approximates K(x) using practical models
- As model class grows, MDL → K(x)

---

## 📊 Real-World Applications

### 1. Polynomial Degree Selection

```python
# Which polynomial degree fits best?
degrees = [1, 2, 3, 5, 10, 20]
best_degree = None
best_mdl = float('inf')

for d in degrees:
    model = PolynomialRegressor(degree=d)
    model.fit(X, y)
    residuals = y - model.predict(X)
    
    # MDL score
    L_H = d * 32  # bits for coefficients
    L_D_H = gaussian_code_length(residuals)
    mdl_score = L_H + L_D_H
    
    if mdl_score < best_mdl:
        best_mdl = mdl_score
        best_degree = d

print(f"Best degree: {best_degree}")
```

### 2. Neural Network Architecture Search

```python
# Which network size is optimal?
architectures = [[32], [64], [128], [64, 32], [128, 64, 32]]

for arch in architectures:
    model = NeuralNet(hidden_dims=arch)
    model.train(X, y)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # MDL score
    L_H = n_params * 32  # bits for weights
    L_D_H = -model.log_likelihood(X, y)
    
    print(f"Arch {arch}: {L_H + L_D_H:.0f} bits")
```

### 3. Time Series Model Selection

```python
# AR(1) vs AR(2) vs AR(5)?
for order in [1, 2, 5]:
    model = AutoRegressive(order=order)
    model.fit(time_series)
    
    # Prequential MDL
    mdl = prequential_mdl(model, time_series)
    print(f"AR({order}): {mdl:.0f} bits")
```

---

## 🎓 The Philosophical Depth

### MDL as Inductive Inference

MDL answers the fundamental question:

> "Given finite data, what should I believe about the world?"

**Answer:** Believe the hypothesis that compresses the data the most.

**Why compression = truth?**

1. **Only true patterns compress**: Random noise is incompressible
2. **Overfitting is expensive**: Memorizing randomness costs more bits than ignoring it
3. **Occam's Razor emerges**: Simpler = shorter = better

### The "No Free Lunch" Perspective

MDL doesn't claim one model is universally best. It says:

> "Given YOUR model class, here's how to pick the best model."

Different model classes → different MDL rankings → domain knowledge matters!

---

## 🔧 What's in This Implementation

```
05_MDL_Principle/
├── README.md                 # This file
├── CHEATSHEET.md             # Quick MDL formulas
├── paper_notes.md            # ELI5 explanation
├── implementation.py         # MDL computation library
├── visualization.py          # MDL score plots
├── train_minimal.py          # Model selection demo
├── notebook.ipynb            # Interactive walkthrough
├── setup.py                  # Environment setup
├── requirements.txt          # Dependencies
├── data/
│   └── input.txt             # Sample datasets
└── Exercises/
    ├── README.md             # Exercise guide
    ├── exercise_01_two_part_code.py
    ├── exercise_02_prequential.py
    ├── exercise_03_model_selection.py
    ├── exercise_04_nml_complexity.py
    ├── exercise_05_mdl_vs_aic_bic.py
    └── solutions.py
```

---

## 🚀 Quick Start

```bash
# Navigate to Day 5
cd papers/05_MDL_Principle

# Install dependencies
pip install -r requirements.txt

# Run the model selection demo
python train_minimal.py

# Or explore interactively
jupyter notebook notebook.ipynb
```

---

## 📈 Expected Output

When you run `train_minimal.py`:

```
==============================================
MDL Model Selection: Polynomial Degree
==============================================

Generating noisy polynomial data (true degree = 3)...
  - 100 data points
  - Noise std: 0.5

Testing polynomial degrees 1-10:

  Degree 1:  L(H)=   64 bits, L(D|H)= 1842 bits, Total= 1906 bits
  Degree 2:  L(H)=   96 bits, L(D|H)=  892 bits, Total=  988 bits
  Degree 3:  L(H)=  128 bits, L(D|H)=  412 bits, Total=  540 bits  ← MDL BEST
  Degree 4:  L(H)=  160 bits, L(D|H)=  401 bits, Total=  561 bits
  Degree 5:  L(H)=  192 bits, L(D|H)=  398 bits, Total=  590 bits
  ...
  Degree 10: L(H)=  352 bits, L(D|H)=  389 bits, Total=  741 bits

✓ MDL correctly identified degree 3!

Comparison with other criteria:
  AIC selected: degree 4
  BIC selected: degree 3
  MDL selected: degree 3

Saved: mdl_model_selection.png
```

---

## 🧩 The 5 Exercises

| # | Topic | What You'll Build |
|---|-------|-------------------|
| 1 | Two-Part Code | Implement basic MDL scoring |
| 2 | Prequential MDL | Sequential prediction coding |
| 3 | Model Selection | Compare polynomials with MDL |
| 4 | NML Complexity | Compute stochastic complexity |
| 5 | MDL vs AIC vs BIC | Head-to-head comparison |

---

## 📚 Key Takeaways

1. **MDL = Compression-based model selection**
   - Best model = shortest total description

2. **Two-Part Code: L(H) + L(D|H)**
   - Model cost + residual cost
   - Simple and intuitive

3. **Prequential Code: Sequential prediction**
   - No arbitrary model description
   - Natural for online learning

4. **NML: The gold standard**
   - Minimax optimal
   - Theoretically beautiful

5. **MDL vs AIC/BIC**
   - MDL adapts penalty to parameter usage
   - More principled, same computational cost

6. **Connection to Kolmogorov Complexity**
   - MDL ≈ practical approximation of K(x)
   - Compression = Understanding

---

## 🔗 Connections to Other Days

| Day | Connection |
|-----|------------|
| Day 4 | Hinton's MDL for neural nets uses these principles |
| Day 3 | Regularization = implicit MDL (shorter weight descriptions) |
| Day 2 | LSTM gates = learned compression of sequences |
| Day 1 | RNN prediction = prequential coding |

---

## 📖 Further Reading

- **Grünwald's Book**: "The Minimum Description Length Principle" (2007) - The definitive reference
- **Rissanen**: "Stochastic Complexity in Statistical Inquiry" - The original inventor
- **Wallace & Boulton**: "An Information Measure for Classification" - MML (related approach)

---

## 🎯 After This Day

You'll understand:
- ✅ Why compression = intelligence (mathematically!)
- ✅ How to select models without arbitrary parameters
- ✅ The deep connection between coding and learning
- ✅ When to use MDL vs AIC vs BIC
- ✅ The philosophical foundation of Occam's Razor

**Tomorrow (Day 6):** We explore complexity itself - what IS complexity, physically?

---

*"The goal of science is to find the shortest description of data that still captures all the regularities."*
— Jorma Rissanen, inventor of MDL
