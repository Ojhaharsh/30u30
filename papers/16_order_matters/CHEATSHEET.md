# Pointer Networks Cheatsheet

**Quick reference for Order Matters: Sequence to Sequence for Sets**

---

## ⏱️ The Big Idea (30 seconds)

**Problem:** How do you process SETS (where order doesn't matter) with neural networks that naturally process SEQUENCES?

**Solution:** 
1. Encode inputs **without** position information (order-invariant)
2. Decode by **pointing** to input elements (not generating from vocabulary)
3. Use attention to select which element to output next

**Result:** Solve problems where input = set, output = sequence (sorting, TSP, convex hull)

---

## 🚀 Quick Start

### Installation

```bash
pip install torch numpy matplotlib scipy
```

### Train a Sorting Model (2 minutes)

```bash
# Clone and navigate
cd papers/16_order_matters

# Train on sorting
python train.py --task sort --set-size 10 --epochs 50

# Visualize results
python visualization.py --checkpoint checkpoints/sort_best.pt
```

### Use in Python

```python
from implementation import ReadProcessWrite

# Create model
model = ReadProcessWrite(
    input_dim=1,          # 1D numbers for sorting
    hidden_dim=128,
    use_set_encoder=True  # Order-invariant!
)

# Sort numbers
inputs = torch.tensor([[[5.0], [2.0], [9.0], [1.0]]])
lengths = torch.tensor([4])
pointers, log_probs, attentions = model(inputs, lengths, max_steps=4)

# pointers = [3, 1, 0, 2] → sorted order!
```

---

## 📐 Core Architecture

### The Three Components

```
INPUT SET
    ↓
[SET ENCODER] ← No positional encoding!
    ↓
ENCODED REPRESENTATIONS
    ↓
[POINTER DECODER] ← Points to inputs
    ↓
OUTPUT SEQUENCE
```

### 1. Set Encoder (Order-Invariant)

```python
class SetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # CRITICAL: No positional encoding!
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.feedforward = nn.Sequential(...)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.self_attention(x, x, x)  # Self-attention
        x = self.feedforward(x)
        return x
```

**Key property:** `encoder([a,b,c]) == encoder([c,a,b])`

### 2. Pointer Attention

```python
class PointerAttention(nn.Module):
    def compute_scores(self, decoder_state, encoder_outputs):
        # Project both to same space
        dec_proj = self.W_decoder(decoder_state).unsqueeze(1)
        enc_proj = self.W_encoder(encoder_outputs)
        
        # Combine and score
        combined = torch.tanh(dec_proj + enc_proj)
        scores = self.v(combined).squeeze(-1)
        
        # Softmax to probabilities
        probs = F.softmax(scores, dim=-1)
        
        # Select pointer
        pointer = torch.argmax(probs, dim=-1)
        return pointer, probs
```

**Formula:** `score_i = v^T * tanh(W_d*s + W_e*h_i)`

### 3. Pointer Decoder

```python
for step in range(max_steps):
    # Update decoder state
    decoder_hidden = self.decoder_cell(prev_input, decoder_hidden)
    
    # Compute attention → get pointer
    pointer, attention = self.attention(decoder_hidden, encoder_outputs)
    
    # Select element and use as next input
    next_input = inputs[pointer]
    
    # Mask out selected element (can't select twice)
    mask[pointer] = 0
```

---

## 📊 Hyperparameter Guide

| Parameter | Small Set | Medium Set | Large Set | Notes |
|-----------|-----------|------------|-----------|-------|
| **Set Size** | 5-10 | 20-30 | 50+ | Start small! |
| **Hidden Dim** | 64-128 | 128-256 | 256-512 | Scale with problem |
| **Num Heads** | 2-4 | 4-8 | 8-16 | More heads = more capacity |
| **Num Layers** | 1-2 | 2-3 | 3-4 | Deeper for harder tasks |
| **Learning Rate** | 1e-3 | 1e-3 | 1e-4 | Reduce for large models |
| **Batch Size** | 128 | 64 | 32 | Reduce if OOM |
| **Dropout** | 0.1 | 0.1-0.2 | 0.2-0.3 | More for large models |
| **Grad Clip** | 2.0 | 2.0 | 1.0 | Prevent exploding gradients |

### Task-Specific Settings

**Sorting:**
- Set size: 5-15 (easy), 20-50 (medium), 100+ (hard)
- Epochs: 20-50
- Usually converges fast!

**Convex Hull:**
- Set size: 10-20 (good balance)
- Epochs: 50-100
- Needs more capacity (hidden_dim=128+)

**TSP:**
- Set size: 10-20 (tractable), 50+ (very hard)
- Epochs: 100-200
- Benefits from deeper models (3-4 layers)

---

## 🔧 Common Issues & Fixes

### Issue 1: Model Doesn't Learn Anything

**Symptoms:**
- Loss doesn't decrease
- Accuracy stays at random (~10% for 10 elements)

**Fixes:**
```python
# ✅ Use teacher forcing during training
pointers, log_probs, _ = model(
    inputs, lengths, max_steps,
    teacher_forcing=targets  # ← Add this!
)

# ✅ Start with smaller sets
set_size = 5  # Not 50!

# ✅ Check loss computation
loss = -log_probs.gather(1, targets).mean()  # Should decrease
```

### Issue 2: NaN Loss

**Symptoms:**
- Loss becomes NaN after few iterations
- Attention weights are all 0 or inf

**Fixes:**
```python
# ✅ Mask BEFORE softmax, use -inf
scores = scores.masked_fill(mask == 0, float('-inf'))
probs = F.softmax(scores, dim=-1)  # Handles -inf correctly

# ✅ Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

# ✅ Check for numerical instability
# Use log_softmax for loss computation
log_probs = F.log_softmax(scores, dim=-1)
```

### Issue 3: Poor Generalization

**Symptoms:**
- Perfect on training set size
- Terrible on larger sets

**Fixes:**
```python
# ✅ Train on variable sizes
set_sizes = [5, 6, 7, 8, 9, 10]
for epoch in range(epochs):
    size = random.choice(set_sizes)
    train_on_size(size)

# ✅ Use curriculum learning
# Start small, gradually increase
if epoch < 20:
    set_size = 5
elif epoch < 40:
    set_size = 10
else:
    set_size = 15
```

### Issue 4: Attention Doesn't Focus

**Symptoms:**
- Attention weights are uniform (all ~0.1 for 10 elements)
- Model selects randomly

**Fixes:**
```python
# ✅ Increase model capacity
hidden_dim = 256  # Instead of 64

# ✅ Use temperature annealing
temperature = max(0.5, 1.0 - epoch * 0.01)
scores = scores / temperature

# ✅ Verify encoder outputs are different
# Print encoder_outputs.std() - should be >> 0
```

### Issue 5: Out of Memory

**Symptoms:**
- CUDA OOM error
- System hangs

**Fixes:**
```python
# ✅ Reduce batch size
batch_size = 32  # or 16, or 8

# ✅ Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(...) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ✅ Reduce model size
hidden_dim = 64
num_layers = 1
```

---

## 💻 Code Snippets

### Creating a Dataset

```python
from torch.utils.data import Dataset

class SortingDataset(Dataset):
    def __init__(self, num_samples, set_size):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size)
        _, sorted_indices = torch.sort(values)
        return values.unsqueeze(1), sorted_indices
    
    def __len__(self):
        return self.num_samples
```

### Training Loop

```python
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Forward
        pointers, log_probs, _ = model(
            inputs, lengths, max_steps,
            teacher_forcing=targets
        )
        
        # Loss: negative log-likelihood
        target_log_probs = log_probs.gather(1, targets)
        loss = -target_log_probs.mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
```

### Evaluation

```python
model.eval()
with torch.no_grad():
    for inputs, targets in val_loader:
        pointers, _, _ = model(inputs, lengths, max_steps)
        
        # Exact match accuracy
        correct = (pointers == targets).all(dim=1).sum()
        accuracy = correct / len(inputs)
```

### Visualizing Attention

```python
import matplotlib.pyplot as plt

def plot_attention(attention_weights, input_values):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Input Elements')
    plt.ylabel('Output Steps')
    plt.xticks(range(len(input_values)), 
               [f'{v:.2f}' for v in input_values])
    plt.title('Pointer Attention Heatmap')
    plt.show()
```

---

## 🎯 Key Formulas

### Pointer Attention Score

$$e_{ij} = v^T \cdot \tanh(W_s \cdot s_{i} + W_h \cdot h_j)$$

Where:
- $s_i$ = decoder state at step $i$
- $h_j$ = encoder output for element $j$
- $W_s, W_h, v$ = learned parameters

### Attention Weights (Softmax)

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

### Pointer Selection

$$p_i = \arg\max_j \alpha_{ij}$$

### Loss (Negative Log-Likelihood)

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log P(y_i | x, y_{<i})$$

---

## 📈 Performance Benchmarks

### Sorting (Set Size = 10)

| Metric | Expected Value |
|--------|---------------|
| Train Accuracy | 99-100% |
| Val Accuracy | 99-100% |
| Convergence | 20-30 epochs |

### Convex Hull (Set Size = 10)

| Metric | Expected Value |
|--------|---------------|
| Train Accuracy | 95-99% |
| Val Accuracy | 90-95% |
| Convergence | 50-100 epochs |

### TSP (Set Size = 10)

| Metric | Expected Value |
|--------|---------------|
| vs Greedy | 5-15% better |
| vs Optimal | 10-30% worse |
| Convergence | 100-200 epochs |

---

## 🔗 Related Concepts

| Concept | Relation | Paper |
|---------|----------|-------|
| **Attention** | Core mechanism | Bahdanau et al. 2014 |
| **Transformers** | USE positional encoding | Vaswani et al. 2017 |
| **Set Transformers** | Modern extension | Lee et al. 2019 |
| **Graph Neural Networks** | Sets with edges | Scarselli et al. 2009 |
| **DETR** | Object detection | Carion et al. 2020 |

---

## 🎓 Quick Reference

### When to Use Pointer Networks

✅ **Use when:**
- Input is a SET (order doesn't matter)
- Output is a SEQUENCE (order matters)
- Output elements come from input set
- Variable-size outputs

❌ **Don't use when:**
- Output from fixed vocabulary (use regular seq2seq)
- Input order matters (use positional encoding)
- Need to generate new tokens (not just select)

### Key Differences from Standard Seq2Seq

| Feature | Standard Seq2Seq | Pointer Networks |
|---------|-----------------|------------------|
| **Positional Encoding** | ✅ Yes | ❌ No (for encoder) |
| **Output Space** | Fixed vocabulary | Input set |
| **Decoder Output** | Generate tokens | Point to inputs |
| **When to Use** | Translation, generation | Sorting, TSP, combinatorial |

---

## 📚 Resources

- **Paper:** https://arxiv.org/abs/1511.06391
- **Original Pointer Networks:** https://arxiv.org/abs/1506.03134
- **Implementation:** `../implementation.py`
- **Training Script:** `../train.py`
- **Exercises:** `../exercises/`

---

**Tip:** Start with sorting on small sets (5-10 elements), then gradually increase complexity! 🚀
