# Day 18 Cheat Sheet: Pointer Networks

Quick reference for the Pointer Network (Vinyals et al., 2015).

---

## The Big Idea (30 seconds)

Standard sequence models select from a **fixed** dictionary. Pointer Networks select from an **input-dependent** dictionary. The attention mechanism is the output layer. By "pointing" directly at input elements, the model can solve combinatorial problems like Convex Hull, Delaunay Triangulation, and TSP where the output vocabulary is exactly the set of input items.

---

## Quick Start

```bash
# Train on sample sorting data
python train_minimal.py --seq_len 5 --epochs 50

# Visualize attention heatmaps
python visualization.py
```

---

## Key Hyperparameters

### Paper's Setup (Section 4.1)
Single layer LSTM, 256 or 512 hidden units, SGD with lr=1.0, batch size 128, weight init uniform [-0.08, 0.08], L2 gradient clipping of 2.0, 1M training examples.

### Our Simplified Implementation

| Parameter | Value | What It Does | Tips |
|:---|:---|:---|:---|
| `input_size` | 1-2 | Features per item | 1 for sorting, 2 for (x, y) coords |
| `hidden_size` | 64-512 | Model capacity | Paper uses 256 or 512 |
| `seq_len` | 5-100 | Input length | Ptr-Nets generalize to N > train length |
| `lr` | 0.001 | Update step size | We use Adam; paper uses SGD with lr=1.0 |

---

## Common Issues & Fixes

### Infinite Loss
```python
# Fix: Ensure correct masking
scores = scores.masked_fill(mask, float('-inf'))
# If you mask EVERY item, softmax returns NaN.
```

### Broadcast Mismatch
```python
# Fix: Unsqueeze the decoder state
decoder_proj = W2(decoder_hidden).unsqueeze(1) # (B, 1, H)
combined = torch.tanh(encoder_proj + decoder_proj) # (B, S, H)
```

---

## The Math (Copy-Paste Ready)

### Pointer Probability (Eq 3)
```python
# 1. Project Query (decoder) and Keys (encoder)
q = self.W2(decoder_hidden).unsqueeze(1)
k = self.W1(encoder_outputs)

# 2. Additive Attention
u = self.v(torch.tanh(q + k)).squeeze(-1)

# 3. Softmax is the output
probs = F.softmax(u, dim=-1)
```

---

## Visualization Examples

```python
from visualization import plot_attention

# Visualize pointer heatmap
# inputs: [0.7, 0.2, 0.9]
# attention_probs: shape (3, 3) 
plot_attention(inputs, attention_probs, save_path='ptr_viz.png')
```

---

## Experiment Ideas

### Easy
- Train on different sequence lengths (5, 10, 15)
- Test generalization: train on 5, test on 10
- Modify temperature in softmax to "sharpen" pointers

### Medium
- Implement greedy vs. sampling decoding
- Add "sampling with replacement" toggle
- Train on a simple TSP dataset with 5 cities

### Advanced
- Implement the "Pointer-Generator" for text copying
- Add Beam Search for better combinatorial optimization
- Compare LSTM encoder vs. Transformer encoder (Self-Attention)

---

## File Reference

| File | Use It For |
|:---|:---|
| `implementation.py` | Core model and attention logic |
| `train_minimal.py` | Quick training loop for sorting |
| `visualization.py` | Plotting heatmaps and pointers |
| `notebook.ipynb` | Interactive build and demo |
| `exercises/` | Practice implementing Eq 3 |

---

## Debugging Checklist

- [ ] Input coordinates are projected into hidden space?
- [ ] Decoder hidden state matches Attention projection size?
- [ ] Masking prevents model from picking padding tokens?
- [ ] Softmax dim is set to -1 (across inputs)?

---

## Pro Tips

1. **Teacher Forcing**: Always use it during training for sequence construction.
2. **Masking**: Critical for TSP; set $u_j = -\infty$ for cities already visited.
3. **Pre-projection**: Raw values should always pass through a Linear layer before the LSTM.

---

## Success Criteria

- Attention heatmaps look like a sharp permutation matrix.
- Sorting accuracy > 95% on sequences of length 5.
- Model generalizes to sequences slightly longer than training data.

---

## Next Day

Day 19: Relational Reasoning

---

**Questions?** Check [README.md](README.md) or ask in the community.
