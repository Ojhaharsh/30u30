# RNN Regularization Cheatsheet 📋

Quick reference for RNN Regularization techniques

---

## The Big Idea (30 seconds)

Regularization prevents overfitting by:
- **Dropout** = Random amnesia during training (forces redundancy)
- **Layer Norm** = Keep everything on the same scale
- **Weight Decay** = Prefer small weights (simpler models)
- **Early Stopping** = Stop before memorizing

---

## The Four Techniques

### 1. Dropout

```
Forward:  out = (x * mask) / keep_prob    # Scale to maintain mean
Backward: dx = (dout * mask)              # Gradient flows through active neurons

Test time: NO dropout (use all neurons)
```

**Code:**
```python
def dropout_forward(x, keep_prob=0.8, training=True):
    if not training:
        return x, None
    mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
    return x * mask, mask
```

### 2. Layer Normalization

```
Forward:  μ = mean(x)
          σ² = var(x)
          x̂ = (x - μ) / sqrt(σ² + ε)
          out = γ * x̂ + β

Backward: Complex chain rule (see implementation.py)
```

**Code:**
```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta
```

### 3. Weight Decay (L2 Regularization)

```
Loss = ModelLoss + (λ/2) * Σ(w²)
Gradient: dW += λ * W
```

**Code:**
```python
def l2_penalty(weights, weight_decay):
    return 0.5 * weight_decay * sum(np.sum(w**2) for w in weights)

total_loss = model_loss + l2_penalty(weights, 0.0001)
```

### 4. Early Stopping

```
if val_loss < best_val_loss:
    best_val_loss = val_loss
    counter = 0
    save_model()
else:
    counter += 1
    if counter >= patience:
        STOP and load best model
```

**Code:**
```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    
    def check(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Continue
        self.counter += 1
        return self.counter < self.patience
```

---

## Quick Start

### Training with Regularization
```bash
python train_minimal.py --dropout 0.2 --weight-decay 0.0001 --patience 5
```

### In Python
```python
from implementation import RegularizedLSTM, EarlyStoppingMonitor

# Create model with dropout and layer norm
model = RegularizedLSTM(
    vocab_size=vocab_size,
    hidden_size=128,
    dropout_keep_prob=0.8,
    use_layer_norm=True
)

# Training loop with early stopping
monitor = EarlyStoppingMonitor(patience=5)
for epoch in range(100):
    train_loss = train_epoch(model, train_data, weight_decay=0.0001)
    val_loss = validate(model, val_data)
    
    if not monitor.check(val_loss, epoch):
        print(f"Stopping at epoch {epoch}")
        break
```

---

## Hyperparameter Guide

| Parameter | Typical Range | Description | Too Low | Too High |
|-----------|--------------|-------------|---------|----------|
| `keep_prob` | 0.5-0.9 | Dropout keep probability | Too much dropout | No regularization |
| `weight_decay` | 0.00001-0.01 | L2 penalty coefficient | No regularization | Weights can't grow |
| `patience` | 3-10 | Early stopping patience | Stops too early | Trains too long |
| `eps` | 1e-5 | Layer norm epsilon | Numerical issues | (rarely a problem) |

### Good Starting Point
```python
keep_prob = 0.8        # 20% dropout
weight_decay = 0.0001  # Light L2
patience = 5           # Wait 5 epochs
use_layer_norm = True  # Always for RNNs
```

---

## Common Issues & Fixes

### 1. Still Overfitting
**Symptom**: Val loss >> train loss

**Fixes:**
```python
# More aggressive dropout
keep_prob = 0.5  # 50% dropout

# Stronger weight decay
weight_decay = 0.001

# More impatient early stopping
patience = 3
```

### 2. Underfitting
**Symptom**: Both losses stay high

**Fixes:**
```python
# Less dropout
keep_prob = 0.9

# Less weight decay
weight_decay = 0.00001

# More patience
patience = 10
```

### 3. Training Unstable (NaN loss)
**Symptom**: Loss explodes to NaN

**Fixes:**
```python
# Add layer normalization
use_layer_norm = True

# Also try gradient clipping
np.clip(gradient, -5, 5)
```

### 4. Slow Convergence
**Symptom**: Takes forever to train

**Fixes:**
```python
# Layer norm often speeds things up
use_layer_norm = True

# Maybe reduce regularization
keep_prob = 0.9
```

---

## Diagnosing Overfitting

### The Gap Rule
```
Gap = train_loss - val_loss

Gap < 0.1:  Great! Keep going
Gap > 0.5:  Warning, starting to overfit
Gap > 1.0:  Overfitting! Increase regularization
```

### Training Curves

**Healthy:**
```
      ↑loss
      │
train │ ──────────____
val   │ ──────────────
      └──────────────→ epochs
```

**Overfitting:**
```
      ↑loss
      │          ↗ val going UP
val   │ ────────/
train │ ──────────────→ train still DOWN
      └──────────────→ epochs
              ↑ STOP HERE!
```

---

## Formulas Quick Reference

### Dropout
$$y = \frac{x \odot m}{p}, \quad m_i \sim \text{Bernoulli}(p)$$

### Layer Norm
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

### Weight Decay
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \|W\|_2^2$$

### Early Stopping
$$\text{stop if } \text{val\_loss}_t > \text{best\_val\_loss} \text{ for } n \text{ consecutive epochs}$$

---

## Recommended Settings by Scenario

### Small Dataset (<10K samples)
```python
keep_prob = 0.5
weight_decay = 0.001
patience = 3
```

### Medium Dataset (10K-100K)
```python
keep_prob = 0.7
weight_decay = 0.0001
patience = 5
```

### Large Dataset (>100K)
```python
keep_prob = 0.9
weight_decay = 0.00001
patience = 10
```

---

## Debugging Checklist

- [ ] Are you applying dropout at test time? (Should NOT!)
- [ ] Is dropout scaled by 1/keep_prob? (Should be!)
- [ ] Is layer norm on the right axis? (axis=-1 usually)
- [ ] Is weight decay applied only to weights? (Not biases!)
- [ ] Are you saving the BEST model? (Not the last!)
- [ ] Is training/validation mode set correctly?

---

*For detailed explanations, see [README.md](README.md). For math derivations, see [paper_notes.md](paper_notes.md).*
