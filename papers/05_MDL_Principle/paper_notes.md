# Paper Notes: A Tutorial Introduction to the Minimum Description Length Principle

## ELI5 (Explain Like I'm 5)

You have data and you want to find the pattern in it. MDL says: the best model is the one that lets you compress the data the most. Not the model with the best fit, and not the simplest model — the best *compressor*.

The intuition: if your model captures real structure in the data, you can describe the data briefly ("here's the pattern, plus these small corrections"). If your model is wrong or too complex, the total description is longer.

This is formalized as: **Total cost = cost to describe the model + cost to describe the data given the model.** Minimize the sum.

Note: This sounds simple, but the paper's real contribution is making it rigorous — showing exactly how to measure "description length" for different model classes, and proving that different approaches (two-part codes, prequential codes, NML) are all valid and connected.

---

## What This Paper Actually Is

**"A Tutorial Introduction to the Minimum Description Length Principle"** by Peter Grünwald (2004). Published in *Advances in Minimum Description Length: Theory and Applications*.

This is a **tutorial/survey paper** — roughly 80 pages covering the theory, philosophy, and practice of MDL. It is not a single-result paper. It systematizes decades of work by Rissanen, Wallace, and others into a coherent framework.

Grünwald later expanded this into a full textbook: *The Minimum Description Length Principle* (MIT Press, 2007).

### What the paper covers

1. **Two-part codes** (crude MDL) — the simplest formulation
2. **Prequential codes** — sequential prediction without explicit model description
3. **Normalized Maximum Likelihood (NML)** — the theoretically optimal universal code
4. **Stochastic complexity** — the NML codelength as a model selection criterion
5. **Connection to Kolmogorov complexity** — MDL as a practical approximation of algorithmic complexity
6. **Comparison with AIC, BIC, cross-validation** — when MDL agrees and disagrees with other criteria
7. **Consistency results** — when and why MDL selects the correct model asymptotically

---

## The Core Framework: Two-Part Codes

### The Idea

To communicate a dataset $D$ to someone, you could send the raw data. Or you could:
1. First send a model $H$ (a pattern or hypothesis)
2. Then send the deviations from that model (residuals)

If the model captures real structure, the total message is shorter than sending raw data.

### The MDL Principle (Two-Part Code Version)

Choose the hypothesis $H$ that minimizes:

$$L(H) + L(D \mid H)$$

where:
- $L(H)$ = bits to describe the model (parameters, structure)
- $L(D \mid H)$ = bits to describe the data given the model (residual errors encoded as a code)

### Why This Prevents Overfitting

- A too-simple model: $L(H)$ is small, but $L(D \mid H)$ is large (poor fit, many residuals to encode)
- A too-complex model: $L(D \mid H)$ is small (good fit), but $L(H)$ is large (many parameters to specify)
- An overfit model (e.g., degree-99 polynomial for 100 points): $L(H)$ can exceed the raw data cost. The "model" is longer than just listing the data.

The sum automatically finds the balance. No tuning parameter is needed.

### Example: Polynomial Degree Selection

Data generated from a degree-3 polynomial with noise. Which degree fits best?

| Degree | L(H) (model bits) | L(D\|H) (residual bits) | Total |
|--------|-------------------|------------------------|-------|
| 1 | 64 | 1842 | 1906 |
| 2 | 96 | 892 | 988 |
| 3 | 128 | 412 | **540** |
| 4 | 160 | 401 | 561 |
| 9 | 320 | 389 | 709 |

MDL selects degree 3 — the true generating degree. The extra parameters in degree 4+ don't compress the data enough to justify their cost.

---

## Encoding Details: How to Count Bits

### Model Description Length L(H)

For a parametric model with $k$ parameters at $b$ bits of precision each:

$$L(H) = k \cdot b$$

This is the crude version. The paper discusses more refined approaches:
- **Universal integer codes** for model order (e.g., polynomial degree)
- **Adaptive precision** — parameters near zero cost fewer bits than large parameters
- **Two-stage codes** — first encode the model class, then the parameters within that class

### Data Description Length L(D|H)

For regression with Gaussian residuals:

$$L(D \mid H) = \frac{n}{2} \log_2(2\pi e \hat{\sigma}^2)$$

where $\hat{\sigma}^2$ is the residual variance. Better fit (smaller $\hat{\sigma}^2$) = fewer bits.

For classification, it's the negative log-likelihood:

$$L(D \mid H) = -\sum_i \log_2 P(x_i \mid H)$$

---

## Beyond Two-Part Codes: Prequential MDL

### The Problem with Two-Part Codes

The two-part code requires choosing how to encode the model, which introduces some arbitrariness (how many bits per parameter? what code for the model structure?). Different encoding choices can lead to different MDL rankings.

### The Prequential Solution

Instead of describing a model and then residuals, predict each data point using only the previous data points:

$$L_{\text{preq}} = \sum_{i=1}^{n} -\log_2 P(x_i \mid x_1, \ldots, x_{i-1})$$

This is the **cumulative log-loss** (surprise) of a sequential predictor. No model description is needed — the model is implicitly defined by the prediction procedure.

**Key property:** This is a valid code length (it satisfies the Kraft inequality), so it's a legitimate MDL score even though it never explicitly describes a model.

**Advantages over two-part codes:**
- No arbitrary model description length
- Natural for online/sequential settings
- The model class's complexity is automatically accounted for through prediction performance

The paper shows that prequential MDL is connected to Bayesian model averaging via the prequential plug-in code, where you use the maximum likelihood estimate at each step.

---

## The Gold Standard: Normalized Maximum Likelihood (NML)

### The Idea

Given a model class $\mathcal{M}$ (e.g., "all polynomials of degree $\leq k$"), define:

$$P_{\text{NML}}(x^n) = \frac{P(x^n \mid \hat{\theta}(x^n))}{\sum_{y^n} P(y^n \mid \hat{\theta}(y^n))}$$

where $\hat{\theta}(x^n)$ is the maximum likelihood estimate for data $x^n$.

### What This Means

The numerator is the best the model class can do on THIS data (maximum likelihood). The denominator sums this over ALL possible datasets — it measures how flexible the model class is.

A more flexible model class has a larger denominator (it can fit many different datasets well), so each individual dataset gets a lower probability — more bits to encode.

### Stochastic Complexity

The NML codelength for data $x^n$ under model class $\mathcal{M}$ is:

$$\text{COMP}(\mathcal{M}, n) = \log \sum_{y^n} P(y^n \mid \hat{\theta}(y^n))$$

This is the **stochastic complexity** of the model class. It measures the "number of different datasets the model class can fit well" — a formal version of model complexity.

**For Gaussian models:**

$$\text{COMP} \approx \frac{k}{2} \log \frac{n}{2\pi}$$

where $k$ is the number of parameters and $n$ is the sample size. This is similar to the BIC penalty but derived from information-theoretic first principles rather than Bayesian approximation.

---

## MDL vs. AIC vs. BIC

The paper devotes significant attention to comparing MDL with other model selection criteria.

| Criterion | Penalty | Derivation | Properties |
|-----------|---------|-----------|------------|
| AIC | $2k$ | KL divergence minimization | Asymptotically efficient; tends to overfit in finite samples |
| BIC | $k \log n$ | Laplace approximation to Bayesian marginal likelihood | Consistent (selects true model as $n \to \infty$) |
| MDL (NML) | $\approx \frac{k}{2} \log \frac{n}{2\pi}$ | Minimax optimal universal code | Consistent; adapts to parameter geometry |

### Key differences from the paper:

1. **AIC does not penalize enough.** For polynomial selection, AIC systematically overfits in small samples because the $2k$ penalty doesn't grow with $n$.

2. **BIC is close to MDL.** For regular exponential family models, BIC approximates the NML codelength. The paper shows the connection: BIC $\approx$ NML + lower-order terms.

3. **MDL adapts to parameter geometry.** Unlike AIC/BIC, which penalize each parameter equally, MDL's penalty depends on how the parameters are used. A parameter near zero (barely used) costs fewer bits than a parameter far from zero.

4. **MDL does not assume a "true model" exists.** AIC and BIC both assume the data was generated by some model in the class. MDL makes no such assumption — it works purely from compression.

---

## Connection to Kolmogorov Complexity

The paper traces MDL's intellectual lineage:

**Kolmogorov Complexity $K(x)$**: The length of the shortest program that produces $x$. This is the "ideal" description length — the absolute limit of compression.

**Problem**: $K(x)$ is uncomputable (undecidable).

**MDL's relationship**: MDL restricts to a computable model class and minimizes description length within that class. As the model class grows (becomes more expressive), MDL codelengths approach $K(x)$:

$$\text{MDL} \to K(x) \text{ as model class } \to \text{ all computable functions}$$

This connection, emphasized by Rissanen, gives MDL a philosophical foundation: learning is compression, and the best compression of a dataset captures its true regularities.

### Important nuance from the paper

Grünwald is careful to distinguish MDL from naive "compression = intelligence" claims. MDL is relative to a model class. Without a good model class, MDL can't find patterns that the class can't express. Domain knowledge matters for choosing the model class.

---

## What the Paper Doesn't Do

- **No deep learning**: The examples are polynomials, Markov models, and simple parametric families. Neural networks are mentioned only in passing.
- **No algorithm for NML computation**: For most model classes, computing the NML normalizing constant is intractable. The paper discusses approximations.
- **No claim that MDL is always better**: Grünwald is explicit that MDL, AIC, and BIC all have regimes where they work well. MDL's advantage is theoretical principled-ness, not necessarily practical superiority in all settings.

---

## Our Implementation (Going Beyond the Paper)

The paper is theoretical. Our implementation makes it concrete:

1. **Two-part code model selection** — polynomial regression with bits computation (demonstrates Section 3 of the paper)
2. **Prequential MDL** — sequential prediction for time-series-like data (implements Section 5)
3. **NML complexity computation** — approximate stochastic complexity for Gaussian models (implements Section 6)
4. **MDL vs. AIC vs. BIC comparison** — head-to-head on polynomial selection (implements Section 8's comparison)
5. **Monte Carlo trials** — statistical comparison across many random datasets (our addition for robustness testing)

The polynomial selection experiment is the paper's running example. The Monte Carlo comparison is our extension.

---

## Key Equations Summary

| Concept | Equation | Source |
|---------|----------|--------|
| Two-Part MDL | $L(H) + L(D \mid H)$ | Paper, Section 3 |
| Gaussian residual code | $\frac{n}{2}\log_2(2\pi e \hat{\sigma}^2)$ | Paper, Section 3 |
| Prequential MDL | $\sum_i -\log_2 P(x_i \mid x_{<i})$ | Paper, Section 5 |
| NML distribution | $P_{\text{NML}}(x) = P(x \mid \hat{\theta}(x)) / C_n$ | Paper, Section 6 |
| Stochastic complexity | $\text{COMP} \approx \frac{k}{2}\log\frac{n}{2\pi}$ | Paper, Section 6 |
| AIC | $-2\log L + 2k$ | Akaike (1974), discussed in Section 8 |
| BIC | $-2\log L + k\log n$ | Schwarz (1978), discussed in Section 8 |

---

## Questions Worth Thinking About

1. The two-part code requires choosing how many bits per parameter. What happens if you use 16-bit vs. 64-bit precision? How does this change which model MDL selects, and why?

2. Prequential MDL depends on the order you process the data. For i.i.d. data this doesn't matter asymptotically, but for small samples it can. What does this imply about using prequential MDL in practice?

3. Grünwald notes that MDL and Bayesian model selection are related but not identical. When do they disagree, and whose answer do you trust?

4. The paper argues that "compression = finding regularities." But a JPEG compresses an image using the DCT, which may have nothing to do with the image's semantic content. Does MDL really capture "understanding," or just statistical regularity?

---

**Next:** [Day 6](../06_Complexodynamics/)
