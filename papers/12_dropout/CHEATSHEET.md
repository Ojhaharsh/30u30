# Dropout Cheatsheet 📋

Quick reference for Preventing Overfitting with Dropout

---

## The Big Idea (30 seconds)

Dropout **randomly disables neurons** during training with probability $p$, forcing the network to learn redundant, robust representations. At test time, all neurons are active (with scaled weights).

- **Training**: Randomly zero out neurons, scale remaining by $1/(1-p)$
- **Testing**: Use all neurons (no scaling with inverted dropout)
- **Result**: Prevents overfitting by acting like an ensemble of networks

**Magic formula**: `keep_prob = 1 - drop_prob`

---

## Architecture: Where to Apply Dropout

### Typical Placement
```
Input Layer (Optional)
    ↓
[Dense Layer]
    ↓
[Activation (ReLU)]
    ↓
[Dropout p=0.5]  ← HERE: After activation
    ↓
[Dense Layer]
    ↓
[Activation (ReLU)]
    ↓
[Dropout p=0.5]  ← HERE: After activation
    ↓
[Output Layer]
    ↓
[Softmax]  ← NO dropout here!
```

### Recommended Dropout Rates
```python
# Input layer (optional noise)
input_dropout = 0.1 to 0.2  # Light dropout

# Hidden layers (main regularization)
hidden_dropout = 0.5  # Standard (50% keep probability)

# Convolutional layers
conv_dropout = 0.1 to 0.25  # Lighter than FC

# Recurrent layers  
rnn_dropout = 0.2 to 0.5  # Variational dropout

# Output layer
output_dropout = 0.0  # NEVER apply dropout here
```

---

## Quick Start

### Basic Dropout Implementation
```python
import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of KEEPING a neuron (not dropping!)
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x  # No dropout during inference
        
        # Generate random mask
        self.mask = (np.random.rand(*x.shape) < self.p) / self.p
        return x * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask
```

### PyTorch Usage
```python
import torch.nn as nn

# Standard dropout
dropout = nn.Dropout(p=0.5)  # p = DROP probability!

# In a model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Applied after activation
        x = self.fc2(x)
        return x

# CRITICAL: Switch modes!
model.train()  # Dropout enabled
model.eval()   # Dropout disabled
```

---

## Key Variants

### 1. Standard Dropout
```python
# Drop individual neurons in fully-connected layers
dropout = nn.Dropout(p=0.5)
```

### 2. Spatial Dropout (Dropout2D)
```python
# Drop entire feature MAPS in CNNs
# Better for convolutional layers!
dropout2d = nn.Dropout2d(p=0.1)

# Drops shape: (batch, channels, 1, 1) → broadcast
```

### 3. Dropout1D
```python
# Drop entire channels in 1D (for sequences)
dropout1d = nn.Dropout1d(p=0.1)
```

### 4. Alpha Dropout (for SELU)
```python
# Maintains self-normalizing property
alpha_dropout = nn.AlphaDropout(p=0.1)
```

### 5. DropConnect
```python
# Drop WEIGHTS instead of activations
class DropConnect(nn.Module):
    def forward(self, x, W):
        if self.training:
            mask = torch.bernoulli(torch.ones_like(W) * self.p)
            W = W * mask / self.p
        return x @ W
```

### 6. DropBlock
```python
# Drop contiguous REGIONS for CNNs
# Better than standard dropout for conv layers
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        self.drop_prob = drop_prob
        self.block_size = block_size
```

---

## Design Patterns

### Dropout + BatchNorm
```python
class Block(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.5):
        super().__init__()
        # Order: Linear → BatchNorm → Activation → Dropout
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout AFTER activation
        return x
```

### Progressive Dropout
```python
class ProgressiveDropoutNet(nn.Module):
    """Increase dropout rate as network gets deeper."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10),
        ])
        # Increasing dropout rates
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2),
            nn.Dropout(0.3),
            nn.Dropout(0.5),
            None,  # No dropout on output
        ])
```

### Scheduled Dropout
```python
def scheduled_dropout(epoch, initial_p=0.5, final_p=0.1, total_epochs=100):
    """Decrease dropout over training (curriculum)."""
    progress = epoch / total_epochs
    p = initial_p - progress * (initial_p - final_p)
    return p
```

---

## Training Tips

### 1. Learning Rate Adjustment
```python
# Dropout adds noise → can often use higher learning rate
lr_without_dropout = 0.001
lr_with_dropout = 0.01  # 2-10x higher

# Or use learning rate warmup
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)
```

### 2. Train Longer
```python
# Dropout slows convergence → train more epochs
epochs_without_dropout = 50
epochs_with_dropout = 100  # ~2x more epochs
```

### 3. Monitor the Gap
```python
def check_overfitting(train_acc, val_acc):
    gap = train_acc - val_acc
    if gap > 0.1:  # 10% gap
        print("Increase dropout!")
    elif gap < 0.02:  # 2% gap
        print("Consider less dropout")
    else:
        print("Good balance!")
```

### 4. Combining with Other Regularization
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),      # BatchNorm
    nn.ReLU(),
    nn.Dropout(0.5),          # Dropout
    nn.Linear(256, 10),
)

# Also add L2 regularization via weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## Common Pitfalls & Solutions

### Issue: Forgetting train/eval mode
```python
# PROBLEM: Dropout active during inference
predictions = model(test_data)  # Wrong if model.training = True!

# SOLUTION: Always switch modes
model.eval()
with torch.no_grad():
    predictions = model(test_data)
model.train()
```

### Issue: Wrong probability interpretation
```python
# PyTorch: p = probability of DROPPING (zeroing)
# Paper: p = probability of KEEPING

# PyTorch dropout(0.5) = keep 50% of neurons
# Be careful when reading papers!
```

### Issue: Dropout on output layer
```python
# BAD: Hurts predictions
model = nn.Sequential(
    nn.Linear(256, 10),
    nn.Dropout(0.5),  # DON'T DO THIS!
    nn.Softmax(dim=1),
)

# GOOD: No dropout before final prediction
model = nn.Sequential(
    nn.Linear(256, 10),
    nn.Softmax(dim=1),
)
```

### Issue: Too much dropout
```python
# Symptoms: Training accuracy stuck at low value
# Training: 55%, Val: 52% → Network can't learn!

# Solution: Reduce dropout rate
dropout_p = 0.8  # Too high!
dropout_p = 0.5  # Better
dropout_p = 0.3  # If still struggling
```

---

## Monte Carlo Dropout (Uncertainty)

```python
def mc_dropout_predict(model, x, n_samples=100):
    """Use dropout at test time for uncertainty estimation."""
    
    model.train()  # Keep dropout active!
    
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    mean = predictions.mean(dim=0)      # Expected prediction
    variance = predictions.var(dim=0)   # Uncertainty
    
    return mean, variance

# Usage
mean, var = mc_dropout_predict(model, test_input)
print(f"Prediction: {mean.argmax()}, Uncertainty: {var.mean():.4f}")
```

---

## Performance Comparison

### When to Use Dropout
```
✅ Fully-connected layers with many parameters
✅ Small to medium datasets (< 100k samples)
✅ When train-val gap is large (overfitting)
✅ Combined with data augmentation
✅ Language model training
```

### When to Avoid/Reduce Dropout
```
❌ Very deep networks (use BatchNorm instead)
❌ Already using heavy data augmentation
❌ Very small networks (not enough capacity)
❌ When training is slow to converge
❌ In ResNet-style skip connections
```

### Modern Best Practices
```python
# For CNNs
conv_layers → BatchNorm + light Dropout2D (0.1)
fc_layers → Dropout (0.5)

# For Transformers
attention → Dropout (0.1)
ffn → Dropout (0.1)
embeddings → Dropout (0.1)

# For very deep networks
Use BatchNorm primarily, minimal dropout
```

---

## Quick Debugging

### Symptom: Training stuck at random accuracy
```python
# Cause: Too much dropout
# Solution: Reduce p from 0.5 to 0.3 or 0.2
```

### Symptom: Large gap but not using dropout
```python
# Cause: Pure overfitting
# Solution: Add dropout 0.5 to hidden layers
```

### Symptom: Val accuracy fluctuates wildly
```python
# Cause: Forgot to call model.eval()
# Solution: Always switch to eval mode for validation
model.eval()
```

### Symptom: Different results each inference
```python
# Cause: Model still in training mode
# Solution: model.eval() before inference
```

---

## Historical Context

### The Paper's Impact
```
Before (2012): Deep networks couldn't generalize
- Train: 99%, Test: 60% = Useless

After (2012): Dropout enables deep learning
- Train: 90%, Test: 85% = Actually works!

AlexNet used dropout → Won ImageNet → Started revolution
```

### Evolution
```
2012: Standard Dropout (Hinton)
2013: DropConnect (Wan)
2015: Variational Dropout for RNNs
2016: Concrete Dropout (learnable rates)
2018: DropBlock for CNNs
2020+: Mostly replaced by BatchNorm for ResNets
       Still crucial for Transformers
```

---

*Dropout: Sometimes the best way to learn is to randomly forget!* 🎲🧠
