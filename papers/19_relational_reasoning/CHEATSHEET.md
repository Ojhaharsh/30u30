# Day 19 Cheat Sheet: Relation Networks

Concise technical reference for the Relation Network (Santoro et al., 2017).

---

## The Big Idea

A **Relation Network (RN)** is a set-processing module that forces a relational inductive bias by bottlenecking information through pairwise interactions. Unlike CNNs (local) or RNNs (sequential), the RN treats inputs as a set and processes all $N^2$ object pairings through a shared non-linear function.

---

## Quick Start
```bash
# Verify environment and permutation invariance
python setup.py

# Train on furthest point task
python train_minimal.py --mode furthest --epochs 30

# Generate relation heatmaps
python visualization.py --num-objects 10
```

---

## Key Technical Relationships

| Measure | Relation Network (RN) | Standard MLP | Transformer (Self-Attn) |
|:---|:---|:---|:---|
| **Input Type** | Unstructured Set | Fixed Vector | Unstructured Set |
| **Inductive Bias** | Relational / Pairwise | None (Global) | Relational / Dynamic |
| **Complexity** | $O(N^2)$ | $O(1)$ | $O(N^2)$ |
| **Order Sensitivity** | Invariant (Sum) | Sensitive | Invariant (w/o Pos Enc) |

---

## Hyperparameters & Dimensions

| Parameter | Symbol | Range | Technical Role |
|:---|:---|:---|:---|
| Object Dim | $d_o$ | 32–128 | Feature vector size per entity |
| Relation Dim | $d_r$ | 256–512 | Hidden width of $g_{\theta}$ |
| Global Dim | $d_f$ | 256–512 | Hidden width of $f_{\phi}$ |
| Dropout | $p$ | 0.5 | Applied in $f_{\phi}$ to prevent overfitting |

---

## Common Issues & Fixes

### 1. Model is Order-Sensitive
**Problem**: The network produces different results if you shuffle the input set.
**Fix**: Ensure your final aggregator is a symmetric function.
```python
# WRONG (Order dependent)
aggregated = g_out.view(batch, -1) 

# CORRECT (Invariant)
aggregated = g_out.sum(dim=1)
```

### 2. OOM (Out of Memory)
**Problem**: GPU memory crashes with large object sets.
**Fix**: Remember that RN is $O(N^2)$. If $N=128$, you are processing 16,384 pairs per batch item. Reduce batch size or number of objects ($N$).

### 3. Fails on Spatial Tasks (CLEVR)
**Problem**: Model cannot reason about "left" or "right."
**Fix**: CNN feature maps don't inherently store absolute coordinates. You must append $(x, y)$ coordinates to each object vector.

---

## Troubleshooting Snips

### Handling $O(N^2)$ Out-Of-Memory
If your object set is too large for your GPU, avoid explicit loops and use mini-batching for relations:
```python
# Instead of processing all N^2 at once:
for i in range(0, num_pairs, sub_batch_size):
    chunk = pairs[i : i + sub_batch_size]
    g_out_chunk = model.g_theta(chunk)
    # Sum into total_rel
```

### Efficient Pair Broadcasting
```python
# Recommended broadcasting logic
objs_i = objects.unsqueeze(2).expand(B, N, N, D)
objs_j = objects.unsqueeze(1).expand(B, N, N, D)
pairs = torch.cat([objs_i, objs_j], dim=-1)
```

---

## Success Criteria

- [ ] `setup.py` passes the permutation invariance check.
- [ ] Model achieves >90% accuracy on the "count" task for $N=10$.
- [ ] Relation heatmaps show high activations for the predicted "furthest" object.
- [ ] `implementation.py` uses broadcasting (no Python loops) for pair generation.

---

## File Reference

| File | Use It For |
|:---|:---|
| `implementation.py` | Core RN class and pairwise broadcasting logic. |
| `train_minimal.py` | Training CLI and relational task variants. |
| `visualization.py` | $g_{\theta}$ heatmaps and spatial distribution plots. |
| `setup.py` | Architectural diagnostics and dependency checks. |

---

**Next:** [Day 20 - Relational Recurrent Neural Networks](../20_relational_rnn/)
