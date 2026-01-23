# MDL & Bayesian NN Cheatsheet 📋

Quick reference for "Keeping Neural Networks Simple" (Hinton 1993)

---

## The Big Idea (30 seconds)

Standard Neural Networks are "Point Estimates"—they learn one specific number for every weight.
MDL (Bayesian) Networks are **"Distribution Estimates"**—they learn a range (Mean ± Uncertainty) for every weight.

* **Standard NN:** "The weight is exactly 5.12." (Brittle, Overconfident)
* **MDL NN:** "The weight is roughly 5.0 ± 0.2." (Robust, Honest)

**The Trade-off:**
We minimize: `Total Loss = Error (NLL) + Complexity (KL)`
* **Error:** Tries to fit the data (makes weights precise).
* **Complexity:** Tries to be simple (makes weights fuzzy).

---

## Architecture: The Bayesian Neuron

Instead of $y = wx + b$, we have:

```python
# 1. Sample a weight from the distribution
w_epsilon ~ N(0, 1)
w = w_mu + w_sigma * w_epsilon

# 2. Compute output
y = w * x + b

```

**Parameters per Weight:**

1. `w_mu`: The center of the weight (the value).
2. `w_rho`: The uncertainty parameter (controls spread).
*  (Softplus)



---

## Quick Start

### Training

```bash
# Train on Gappy Sine Wave (Default)
python train_minimal.py

# Strong Regularization (Simpler Model)
python train_minimal.py --kl-weight 0.5

# Weak Regularization (Complex Model)
python train_minimal.py --kl-weight 0.001

```

### In Python

```python
from implementation import MDLNetwork

# Create Network
net = MDLNetwork(input_size=1, hidden_size=20, output_size=1)

# Forward pass (Returns DIFFERENT result every time!)
pred_1 = net.forward(x)
pred_2 = net.forward(x)

# Calculate Loss
loss = mse_loss(pred_1, y) + kl_weight * net.total_kl()

```

---

## Hyperparameter Guide

| Parameter | Typical Range | Description | Too Low | Too High |
| --- | --- | --- | --- | --- |
| `kl_weight` (Beta) | 0.001 - 1.0 | The "Price" of complexity. **Crucial.** | Overfits (Standard NN behavior) | Underfits (Flat line prediction) |
| `hidden_size` | 10 - 50 | Number of neurons. | Can't fit the curve | Hard to train (too much noise) |
| `rho_init` | -3.0 to -5.0 | Initial uncertainty. | Starts too precise (gets stuck) | Starts too noisy (diverges) |
| `lr` | 0.001 - 0.01 | Learning rate. | Slow convergence | Unstable gradients |

### Good Starting Point

```python
hidden_size = 20
kl_weight = 0.1  # Adjust this if model ignores data!
lr = 0.01
epochs = 2000

```

---

## Common Issues & Fixes

### 1. The "Flat Line" Problem (Underfitting)

**Symptom:** The model predicts a straight line at y=0 and ignores the data.
**Cause:** `kl_weight` is too high. The model is terrified of being complex, so it learns nothing.
**Fix:**

* Reduce `kl_weight` (try 0.1 → 0.01).
* Increase data size (data overpowers the prior).

### 2. The "Overconfident" Problem (Overfitting)

**Symptom:** The Uncertainty Envelope is tiny, even where there is no data.
**Cause:** `kl_weight` is too low. The model is effectively a Standard NN.
**Fix:**

* Increase `kl_weight`.
* Check if `total_kl()` is being scaled correctly (should be divided by N_samples).

### 3. Training Diverges (NaN)

**Symptom:** Loss explodes.
**Cause:** The variance () exploded or became 0.
**Fix:**

* Use `softplus` for sigma (ensures ).
* Clip gradients.
* Lower learning rate.

---

## Visualization Guide

```python
from visualization import plot_uncertainty_envelope, plot_weight_distributions

# 1. The Money Shot (Uncertainty)
plot_uncertainty_envelope(net, X_train, y_train)
# Look for: Wide bubbles in the gaps, tight fit on data.

# 2. The Compression Check
plot_weight_distributions(net, 'layer1')
# Look for: Many weights with High Sigma (The "Prunable" ones).

```

---

## Comparison: Standard vs Bayesian

| Aspect | Standard NN (SGD) | MDL / Bayesian NN |
| --- | --- | --- |
| **Weights** | Fixed Numbers | Distributions (Mean, Std) |
| **Prediction** | Always the same | Different every time (Sampling) |
| **Unknown Data** | "I am 100% sure" (Wrong) | "I don't know" (High Variance) |
| **Pruning** | Hard (Magnitude based) | Natural (Sigma based) |
| **Math** | Calculus (Chain Rule) | Calculus + Statistics (KL Div) |

---

## Key Equations

```text
Weight Sample:    w = μ + σ * ε      where ε ~ N(0,1)
Prediction:       y = w * x + b

Complexity Cost:  KL = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
Error Cost:       NLL ≈ MSE / (2 * noise_variance)

Total Loss:       L = NLL + β * KL

```

**Remember:**

* **KL** pulls weights towards 0 and makes them fuzzy.
* **NLL** pulls weights towards data and makes them precise.
* Training is a tug-of-war between these two.

---

## Resources

* **Paper:** "Keeping Neural Networks Simple" (Hinton 1993)
* **Code:** `implementation.py`
* **Visuals:** `visualization.py`
* **Modern Version:** "Bayes by Backprop" (Blundell 2015)

---

That's it! You're now ready to go ahead like a pro! 🚀

**Next**: Try exercises 1-5 to solidify your understanding.