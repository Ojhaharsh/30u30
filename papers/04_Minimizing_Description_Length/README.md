# Day 4: Keeping Neural Networks Simple (MDL)

> Hinton & van Camp (1993) — [COLT '93 paper](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)

**Time:** 3-5 hours
**Prerequisites:** Day 3 (regularization concepts)
**Code:** NumPy only

---

## What This Paper Is Actually About

This is a 6-page theoretical paper that connects three ideas:

1. **Minimum Description Length (MDL)**: The best model minimizes total bits to describe model + data errors
2. **Bayesian neural networks**: Weights are probability distributions, not fixed numbers
3. **Variational inference**: Optimize a tractable lower bound on the true posterior

Hinton and van Camp show these are **the same thing**. If you represent weights as Gaussian distributions and minimize description length using "bits back" coding, you end up minimizing the variational free energy:

$$\mathcal{L} = \underbrace{E_q[\text{error}]}_{\text{data fit}} + \underbrace{KL(q \| p)}_{\text{complexity cost}}$$

In 1993, this connection was not obvious. The paper is the theoretical foundation for much of Bayesian deep learning that followed.

---

## The Core Idea

### Standard Weights vs. Noisy Weights

Standard neural network: each weight is a fixed number (point estimate).

```
Standard:  w = 5.123456    (one number, high precision, many bits)
MDL:       w ~ N(5.1, 0.2) (a distribution — mean and spread)
```

The MDL approach asks: if the network still works when the weight is noisy, you didn't need all that precision. The saved precision translates directly to saved description bits.

### The "Bits Back" Argument

This is the paper's key insight. When you encode weights using a distribution rather than a point value:

1. Specifying a sample from distribution $q(w)$ costs $-\log q(w)$ bits
2. But the receiver can reconstruct the randomness — getting bits "back"
3. Against a shared prior $p(w)$, the net cost per weight is $\log \frac{q(w)}{p(w)}$
4. In expectation: $E_q[\log q/p] = KL(q \| p)$

The KL divergence between posterior and prior IS the description cost. This is how an information-theoretic argument produces the same objective as variational Bayes.

### The Full Objective

$$\mathcal{L} = \underbrace{KL(q(w) \| p(w))}_{\text{weight cost}} + \underbrace{E_{q(w)}[-\log p(D|w)]}_{\text{error cost}}$$

- **KL term** pulls weights toward the prior (simpler, fuzzier)
- **Error term** pulls weights toward fitting the data (more precise)
- Training is a tug-of-war between these two forces

### The KL Divergence (Closed Form)

For posterior $q = \mathcal{N}(\mu, \sigma^2)$ and prior $p = \mathcal{N}(0, 1)$:

$$KL = \frac{1}{2}\sum_i(\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2)$$

Properties:
- Large $|\mu|$ is penalized (weights pulled toward zero)
- Small $\sigma$ is penalized (must maintain some uncertainty)
- Very large $\sigma$ is penalized by $\sigma^2$ (can't be infinitely noisy)

---

## Why This Works: Flat Minima

The noise in the weights means the network **can't sit in a sharp minimum** of the loss landscape. If the loss changes drastically when a weight jiggles by $\sigma$, the expected error is high. The network is forced to find wide, flat regions where the loss is stable despite weight noise.

Flat minima tend to generalize better — small shifts from training to test data don't cause large error increases. This connection between flat minima and generalization is now well-established (Hochreiter & Schmidhuber 1997, Keskar et al. 2017), and this paper is one of the earliest to make the argument.

---

## What the Paper Showed

The paper includes a proof-of-concept experiment on a simple regression task:
- A two-layer network learns a target function
- The MDL approach finds simpler solutions that generalize better
- The description length trade-off (complexity vs. accuracy) works as predicted

This is a **theory paper**, not a benchmark paper. There are no MNIST results, no state-of-the-art claims. The contribution is the mathematical framework.

---

## The Architecture

### Standard vs. Bayesian Neuron

```
Standard Neuron:                Bayesian Neuron:
  w = fixed value                 w ~ N(mu, sigma^2)
  y = w*x + b                    w = mu + sigma * epsilon   (epsilon ~ N(0,1))
  Same output every time          y = w*x + b
                                  Different output each forward pass!
```

### Forward Pass (Our Implementation)

```python
# For each weight, we store mu and rho (where sigma = softplus(rho))
sigma = log(1 + exp(rho))         # Ensure sigma > 0
epsilon = sample from N(0, 1)     # Random noise
w = mu + sigma * epsilon          # "Reparameterization trick" (Kingma 2014)

# Standard forward pass with the sampled weights
y = x @ w + b
```

Note: The reparameterization trick ($w = \mu + \sigma \cdot \epsilon$) is from Kingma & Welling (2014), not this paper. We use it because it enables gradient-based training through the sampling step.

### Training Loop

```python
for epoch in range(epochs):
    # Forward: sample weights and compute prediction
    pred = net.forward(X)           # Weights are sampled internally

    # Loss = Error + beta * Complexity
    error = MSE(pred, y)            # How well does it fit?
    complexity = net.total_kl()     # How complex are the weights?
    loss = error + beta * complexity

    # Backward and update
    net.backward(d_loss)
    net.update_weights(lr, beta)
```

### Prediction with Uncertainty

```python
# Run the model N times — each time samples different weights
predictions = [net.forward(X_test) for _ in range(100)]

mean = average(predictions)        # Best estimate
std = standard_deviation(predictions)  # Uncertainty
```

Where the network has seen data: low uncertainty (tight predictions).
Where the network hasn't seen data: high uncertainty (spread-out predictions).

---

## Training Hyperparameters

The paper doesn't specify practical training details (it's a theory paper). These are our recommended settings for the implementation:

| Parameter | Typical Range | What It Controls |
|-----------|--------------|------------------|
| `kl_weight` ($\beta$) | 0.001 - 1.0 | Trade-off: data fit vs. simplicity. Most important parameter. |
| `hidden_size` | 10 - 50 | Network capacity |
| `rho_init` | -3.0 to -5.0 | Initial uncertainty (softplus(-3) ~ 0.05) |
| `lr` | 0.001 - 0.01 | Learning rate |

### Starting point
```python
hidden_size = 20
kl_weight = 0.1
lr = 0.01
epochs = 2000
```

---

## Implementation Notes

Our code in `implementation.py` goes beyond the paper:

- **Reparameterization trick**: From Kingma & Welling (2014), not this paper. Enables backprop through weight sampling.
- **Softplus parameterization**: We learn $\rho$ and compute $\sigma = \log(1 + e^\rho)$ to guarantee $\sigma > 0$.
- **Monte Carlo uncertainty**: Multiple forward passes to estimate prediction intervals. Not in the paper but standard practice.
- **Gappy sine wave experiment**: Our pedagogical choice to demonstrate epistemic uncertainty visually.

Key things to watch for:
- **Every forward pass is different** — weights are sampled each time. This IS the point.
- **$\beta$ (kl_weight) is critical** — too high and the model ignores data (flat line). Too low and it overfits (acts like a standard NN with no uncertainty).
- **Gradients come from two sources** — the error loss (through the sampled weights) AND the KL divergence (directly from $\mu$ and $\sigma$).

---

## What to Build

### Quick Start

```bash
python train_minimal.py
python train_minimal.py --kl-weight 0.5    # More regularization
python train_minimal.py --kl-weight 0.001  # Less regularization
```

### Exercises (in `Exercises/`)

| # | Task | What You'll Learn | Source |
|---|------|-------------------|--------|
| 1 | Reparameterization trick | Sampling weights differentiably | Our addition (technique from Kingma 2014) |
| 2 | Gap experiment | Epistemic uncertainty visualization | Our addition |
| 3 | Beta parameter study | The complexity-accuracy trade-off | Our addition (explores the paper's core equation) |
| 4 | Monte Carlo predictions | Aggregating stochastic predictions | Our addition |
| 5 | Pareto frontier | Compression vs accuracy analysis | Our addition (directly visualizes the MDL principle) |

All exercises are our pedagogical additions — the paper itself is theoretical with a minimal experiment. Exercises 3 and 5 are most directly connected to the paper's core ideas.

Solutions are in `solutions.py` and `solutions_extra.py`. Try first.

---

## Key Takeaways

1. **MDL = Variational Bayes.** The paper's central contribution: minimizing description length of weights produces the same objective as variational inference. Weight cost = KL divergence.

2. **Noisy weights find flat minima.** By training with noise in the weights, the network is forced into wide, stable regions of the loss landscape. These generalize better because small perturbations don't cause large error changes.

3. **Uncertainty is a feature.** Bayesian weights naturally indicate what the network doesn't know — predictions vary more where data is sparse. Standard networks are confident everywhere, even where they shouldn't be.

4. **Simplicity has a cost.** The $\beta$ parameter controls the trade-off explicitly. There's no free lunch — simpler models are more robust but less precise.

---

## Historical Significance

This paper is one of the earliest to connect information theory (MDL/compression) with Bayesian treatment of neural networks. It preceded:
- Variational Autoencoders (Kingma & Welling 2014) — same ELBO objective
- Bayes by Backprop (Blundell et al. 2015) — modern implementation of these ideas
- MC Dropout (Gal & Ghahramani 2016) — showed dropout approximates variational inference

MacKay (1992) independently developed a Bayesian framework for neural networks around the same time, using Laplace approximation rather than variational methods. Together, these two lines of work established the field of Bayesian deep learning.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Bayesian neural network in NumPy: BayesianLinear layer + MDLNetwork |
| `train_minimal.py` | Training script: gappy sine wave experiment with CLI args |
| `visualization.py` | Uncertainty envelopes, weight distributions, loss curves |
| `notebook.ipynb` | Interactive walkthrough of MDL concepts |
| `Exercises/` | 5 exercises: reparameterization, gap experiment, beta study, MC sampling, Pareto frontier |
| `paper_notes.md` | Detailed notes on the actual Hinton & van Camp paper |
| `CHEATSHEET.md` | Quick reference for equations and hyperparameters |

---

## Further Reading

- [Hinton & van Camp (1993)](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) — this paper
- [MacKay (1992)](http://www.inference.org.uk/mackay/thesis.pdf) — parallel Bayesian NN framework using Laplace approximation
- [Blundell et al. (2015)](https://arxiv.org/abs/1505.05424) — "Weight Uncertainty in Neural Networks" (modern version of this paper)
- [Kingma & Welling (2014)](https://arxiv.org/abs/1312.6114) — VAEs, which use the same ELBO + reparameterization trick
- [Gal & Ghahramani (2016)](https://arxiv.org/abs/1506.02142) — MC Dropout as approximate variational inference

---

**Next:** [Day 5](../05_Pointer_Networks/)
