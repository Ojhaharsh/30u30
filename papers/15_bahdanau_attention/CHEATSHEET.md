# Bahdanau Attention Cheatsheet 📋

Quick reference for Neural Machine Translation with Attention

---

## The Big Idea (30 seconds)

Attention solves the **bottleneck problem** by letting the decoder look back at all encoder states instead of cramming everything into one vector. Think of it as:

- **Without attention** = Read a book, close it, translate from memory 😵
- **With attention** = Keep the book open, look up words as needed! 📖🔦

---

## One-Liner
> "Let the decoder look at different parts of the input at each step."

## The Problem
Traditional seq2seq compresses entire input into ONE fixed vector.
Long sentences? Information gets lost. 😵

## The Solution: Attention!

```
Instead of: Encode → Fixed Vector → Decode
Do this:    Encode → All Vectors → Attend → Decode
```

## Core Equations

### 1. Attention Score (Additive/Bahdanau)
```
e_ij = v^T · tanh(W_s · s_{i-1} + W_h · h_j)
```
- `s_{i-1}`: decoder state (query)
- `h_j`: encoder state (key)
- `e_ij`: how relevant is input j for output i?

### 2. Attention Weights
```
α_ij = softmax(e_i) = exp(e_ij) / Σ exp(e_ik)
```
- Weights sum to 1
- Higher weight = more attention

### 3. Context Vector
```
c_i = Σ α_ij · h_j
```
- Weighted sum of encoder outputs
- Different for each decoder step!

### 4. Decoder Update
```
s_i = GRU(s_{i-1}, [y_{i-1}; c_i])
y_i = softmax(W_o · [s_i; c_i; y_{i-1}])
```

## PyTorch Implementation

```python
# Attention Score
class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        self.W_h = nn.Linear(enc_dim, attn_dim)  # Key projection
        self.W_s = nn.Linear(dec_dim, attn_dim)  # Query projection
        self.v = nn.Linear(attn_dim, 1)          # Score projection
    
    def forward(self, decoder_state, encoder_outputs, mask=None):
        # decoder_state: (batch, dec_dim)
        # encoder_outputs: (batch, src_len, enc_dim)
        
        scores = self.v(torch.tanh(
            self.W_h(encoder_outputs) + 
            self.W_s(decoder_state).unsqueeze(1)
        )).squeeze(-1)  # (batch, src_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -inf)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        
        return context.squeeze(1), weights
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                    ENCODER                          │
│  [Bidirectional GRU: sees past AND future]          │
│  Input: x₁, x₂, ..., xₙ                             │
│  Output: h₁, h₂, ..., hₙ (all states kept!)         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                   ATTENTION                         │
│  For each decoder step i:                           │
│  1. Score each encoder state: e_ij = align(s, h_j)  │
│  2. Normalize: α_ij = softmax(e_i)                  │
│  3. Weighted sum: c_i = Σ α_ij · h_j                │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                    DECODER                          │
│  [GRU with attention context]                       │
│  Input: previous token + context vector             │
│  Output: next token probabilities                   │
└─────────────────────────────────────────────────────┘
```

## Key Differences: Bahdanau vs Luong

| Aspect | Bahdanau (Additive) | Luong (Multiplicative) |
|--------|---------------------|------------------------|
| Score | v^T·tanh(Ws + Wh) | s^T · W · h or s^T · h |
| When | BEFORE GRU step | AFTER GRU step |
| Speed | Slower (more params) | Faster |
| Power | More expressive | Simpler |

## Training Tips

```python
# 1. Use padding mask
mask = (src == PAD_IDX)
context, weights = attention(state, encoder_out, mask)

# 2. Teacher forcing
if random.random() < teacher_forcing_ratio:
    next_input = target[t]  # Ground truth
else:
    next_input = prediction.argmax()  # Model's guess

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Ignore padding in loss
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

## Visualization: What Good Attention Looks Like

For sequence reversal [A, B, C, D] → [D, C, B, A]:

```
        Input
        A    B    C    D
    D  [0.1  0.1  0.1  0.8]  ← Looks at D
Out C  [0.1  0.1  0.8  0.1]  ← Looks at C
    B  [0.1  0.8  0.1  0.1]  ← Looks at B
    A  [0.8  0.1  0.1  0.1]  ← Looks at A

= Reversed diagonal pattern! ✓
```

---

## Quick Start

### Setup
```bash
# Install dependencies
pip install torch numpy matplotlib

# Verify setup
python setup.py

# Run tests
python -m pytest solutions/
```

### Training
```bash
# Train on toy reversal task
python train.py

# With custom hyperparameters
python train.py --epochs 50 --hidden-size 256 --batch-size 64

# Quick test run
python train.py --epochs 5 --num-samples 1000
```

### Visualization
```bash
# Demo visualization (no training needed)
python visualization.py --demo

# Visualize trained model
python visualization.py --model checkpoints/best_model.pt
```

### In Python
```python
from implementation import Seq2SeqWithAttention

# Create model
model = Seq2SeqWithAttention(
    src_vocab_size=50,
    trg_vocab_size=50,
    embed_size=64,
    hidden_size=128
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
outputs, attentions = model(src, src_lengths, trg)
loss = criterion(outputs.view(-1, vocab_size), trg[:, 1:].view(-1))

# Inference
translations, attention_weights = model.translate(src, src_lengths)
```

---

## Hyperparameter Guide

| Parameter       | Typical Range | Description                        | Too Low              | Too High             |
|-----------------|---------------|------------------------------------|-----------------------|----------------------|
| `hidden_size`   | 64-512        | Size of RNN hidden states          | Can't capture patterns | Overfits, slow       |
| `embed_size`    | 32-256        | Word embedding dimension           | Poor word repr.       | Wasteful             |
| `attention_dim` | 64-256        | Attention hidden layer             | Weak alignments       | Overfits             |
| `learning_rate` | 0.0001-0.01   | Step size for updates              | Slow learning         | Unstable             |
| `batch_size`    | 16-128        | Samples per gradient update        | Noisy gradients       | Memory issues        |
| `epochs`        | 10-100        | Full passes through dataset        | Underfits             | Overfits             |
| `dropout`       | 0.1-0.5       | Regularization strength            | May overfit           | Loses information    |
| `teacher_forcing`| 0.5-1.0      | Use ground truth in training       | Slow convergence      | Exposure bias        |

### Good Starting Point
```python
hidden_size = 128
embed_size = 64
attention_dim = 128
learning_rate = 0.001
batch_size = 32
epochs = 30
dropout = 0.1
teacher_forcing_ratio = 1.0  # Full teacher forcing
```

---

## Common Issues & Fixes

### 1. Loss Explodes (NaN)
**Symptom**: Loss becomes `nan` after few iterations

**Causes & Fixes**:
```python
# Reduce learning rate
learning_rate = 0.0001  # instead of 0.01

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for -inf in attention before softmax
if mask is not None:
    scores = scores.masked_fill(mask, float('-inf'))
```

### 2. Attention Weights Don't Sum to 1
**Symptom**: `weights.sum(dim=-1)` ≠ 1.0

**Fix**:
```python
# Make sure softmax is on the right dimension!
weights = F.softmax(scores, dim=-1)  # NOT dim=1 or dim=0!
```

### 3. Model Doesn't Learn (Loss Flat)
**Symptom**: Loss stays high, no improvement

**Fixes**:
```python
# Check encoder output shapes
print(f"Encoder outputs: {encoder_outputs.shape}")  # Should be (batch, src_len, hidden)

# Verify gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")

# Try without masking first (debug)
context, weights = attention(decoder_state, encoder_outputs, mask=None)
```

### 4. Poor Attention Patterns
**Symptom**: Attention is uniform (0.25, 0.25, 0.25, 0.25)

**Fixes**:
```python
# Train longer - attention takes time to learn
epochs = 50  # instead of 10

# Check attention dimension is large enough
attention_dim = 128  # instead of 32

# Verify encoder states have variety
print(encoder_outputs.std())  # Should not be ~0
```

### 5. Translation Repeats or Stops Early
**Symptom**: Output is "the the the the" or stops after 2 words

**Fixes**:
```python
# Check EOS/SOS handling
if output_token == eos_idx:
    break  # Stop generating

# Lower temperature during inference for more focused outputs
# Or increase temperature if too repetitive
```

---

## Common Bugs & Fixes (Quick Reference)

| Bug | Symptom | Fix |
|-----|---------|-----|
| Wrong mask dim | Crashes | Ensure (batch, src_len) |
| Missing unsqueeze | Broadcast error | Check broadcasting |
| Softmax wrong dim | Weights don't sum to 1 | Use `dim=-1` |
| No gradient clip | Exploding gradients | Add clip_grad_norm |
| Wrong hidden init | Poor convergence | Use encoder final state |
| Packed sequence issues | Runtime error | Sort by length, use enforce_sorted=False |

---

## Paper Citation

```bibtex
@article{bahdanau2014neural,
  title={Neural Machine Translation by Jointly Learning to Align and Translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1409.0473},
  year={2014}
}
```

## Key Takeaways

1. **Attention solves the bottleneck** - No more single fixed vector
2. **Alignment is learned** - Network decides what to look at
3. **Interpretable** - Visualize attention weights
4. **Foundation for Transformers** - Self-attention builds on this

---

*"The ability to attend to different parts of the input sequence is
what makes this model powerful."* — Bahdanau et al.
