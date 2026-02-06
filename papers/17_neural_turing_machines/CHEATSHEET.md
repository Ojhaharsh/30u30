# Day 17 Cheat Sheet: Neural Turing Machines (NTM)

Quick reference for training and monitoring your NTM.

---

## Quick Start

```bash
# Train on the standard Copy Task
python train.py

# Visualize results (requires a trained model)
# python visualization.py --checkpoint ntm_copy_final.pth
```

---

## Key Hyperparameters (Graves et al., 2014)

| Parameter | Typical Value | What It Does | Rationale (from Paper) |
|-----------|---------------|--------------|-------------------------|
| `memory_n` | 128 | Number of addressable slots | Scale independently of weights (Sec 3) |
| `memory_m` | 20 | Dimension of each slot | Enough precision for bitmask storage |
| `controller_h` | 100 | Size of RNN hidden state | The "thinking" capacity (Sec 2) |
| `learning_rate`| 1e-4 | RMSProp step size | Sec 4: RMSProp is recommended for NTM |
| `grad_clip` | 10.0 | Maximum gradient norm | Sec 4: Essential for training stability |

---

## Common Issues & Fixes

### Loss is "NaN"
- **Reason:** Gradients exploded during the circular convolution step (Eq 8).
- **Fix:** Ensure `nn.utils.clip_grad_norm_` is set to 10.0 or lower in `train.py`.

### Bit Error Rate (BER) doesn't decrease
- **Reason:** The model failed to learn the "Sharpening" mechanism (Eq 9), resulting in blurry addresses.
- **Fix:** Check that your `gamma` parameter is properly initialized and constrained to $\gamma \ge 1$ using `1 + F.softplus`.

### Memory focus jumps randomly
- **Reason:** The Interpolation gate (Eq 7) is not learning to steady the focus.
- **Fix:** Increase the training iterations; NTMs often take 10,000+ iterations to "stabilize" their pointer logic.

---

## The Addressing Equations

1. **Content:** $w_t^c = \text{Softmax}(\beta_t \cdot \text{CosineSim}(k_t, M_t))$
2. **Gating:** $w_t^g = g_t w_t^c + (1 - g_t) w_{t-1}$
3. **Shift:** $w_t^s = w_t^g * s_t$ (Circular Convolution)
4. **Sharp:** $w_t = (w_t^s)^\gamma / \sum (w_t^s)^\gamma$
