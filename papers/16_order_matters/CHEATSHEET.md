# Day 16 Cheat Sheet: Sequence to Sequence for Sets

Quick reference for training and deploying Pointer Networks.

---

## Quick Start

```bash
# Train on the Sorting Task (default)
python train.py --task sort --set-size 10

# Train on Convex Hull
python train.py --task hull --set-size 20

# Visualize results
# python visualization.py --checkpoint checkpoints/model.pt
```

---

## Key Hyperparameters (Vinyals et al., 2015)

| Parameter | Typical Value | What It Does | Rationale (from Paper) |
|-----------|---------------|--------------|-------------------------|
| `embedding_dim`| 128 | Vector size for set elements | Sufficient for coordinates/scalars |
| `hidden_dim` | 256 | LSTM state size | Captures global set context |
| `processing_steps` | 3-5 | Steps of "thinking" | Sec 3: Improves combinatorial search |
| `learning_rate`| 1e-3 | Adam/RMSProp lr | Sec 4: Standard for pointer experiments |
| `grad_clip` | 2.0 | Gradient threshold | Essential for sequential stability |

---

## Common Issues & Fixes

### Model selects the same element twice
- **Reason:** The attention mechanism didn't learn to ignore previously selected indices.
- **Fix:** Implement **Hard Masking**. Set the attention scores of selected indices to `-inf` before the softmax step in `implementation.py`.

### Performance drops on larger sets
- **Reason:** The model overfit to the fixed sequence length used in training.
- **Fix:** Use **Variable Length Training**. Train on a range of set sizes (e.g., sample $N \in [5, 15]$ per batch) to force the model to learn the selection rule rather than a fixed pattern.

### Loss becomes NaN
- **Reason:** Dividing by zero in attention scores or exploding gradients in the LSTM.
- **Fix:** Use `log_softmax` for NLL loss and ensure `torch.nn.utils.clip_grad_norm_` is active.

---

## Core Formulas

1. **Pointer Attention Score:** $u_i = v^T \tanh(W_1 e_i + W_2 d)$
2. **Probability Distribution:** $P(p | e_1, \dots, e_n, d) = \text{Softmax}(u)$
3. **Read Operation (Set-to-Set):** $h = \text{SelfAttention}(X)$ (Equivalent to the "Read" phase)
