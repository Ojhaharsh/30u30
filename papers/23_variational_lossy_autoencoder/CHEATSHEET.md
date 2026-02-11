# Day 23 Cheat Sheet: Variational Lossy Autoencoder

## The Big Idea (30 seconds)

VAEs with powerful autoregressive decoders (like PixelCNN) suffer from **posterior collapse** — the model ignores the latent code $z$ because local pixel prediction is "easier." **VLAE** fixes this by:
1.  **Limiting the decoder's receptive field** (forcing it to look at $z$ for global structure).
2.  **Using a Flow prior (IAF)** (making the distribution $p(z)$ flexible enough to match the posterior).

**Cure the collapse:** Force the model to use $z$ by making it necessary for global coherence.

---

## Quick Start (Training & Sampling)

```bash
# Verify Implementation (Unit Tests)
python setup.py

# Train VLAE on MNIST
python train_minimal.py --epochs 10 --use-flow

# Qualitative Check (Reconstructions)
# See output plots in results/reconstructions.png
```

---

## Architecture at a Glance

| Component | Choice | Why? |
|-----------|--------|------|
| **Encoder** | ResNet-style | Compresses image to $\mu, \sigma$ for $z$. |
| **Prior $p(z)$** | **IAF (Inverse Autoregressive Flow)** | Allows prior to learn complex structure (Eq 5). |
| **Decoder $p(x|z)$** | **Restricted Gated PixelCNN** | Handles local texture; blind to global patterns. |
| **Training** | **Teacher Forcing** | Feed original pixel $x_i$ to predict $x_{i+1}$. |

---

## Key Hyperparameters

| Parameter | Range | Importance |
|-----------|-------|------------|
| `latent_dim` | 16 - 64 | **High**: Global information capacity. |
| `decoder_layers`| 2 - 4 | **CRITICAL**: Controls receptive field (Keep < 8). |
| `kernel_size` | 3 or 5 | Controls how quickly receptive field grows. |
| `flow_steps` | 2 - 4 | Complexity of the prior $p(z)$. |

---

## Common Issues & Fixes

- **KL Divergence is 0:** Classic posterior collapse. **Fix:** Decrease decoder layers or kernel size. 
- **Checkerboard Artifacts:** Often due to bad latent injection or masking bias. **Fix:** Ensure $z$ is upsampled correctly and used in every Gated layer.
- **Blurry Images:** VAE tendency. **Fix:** Increase PixelCNN layers (but don't exceed the global structure threshold).

---

## The Math (Copy-Paste Ready)

### Inverse Autoregressive Flow (IAF)
$$ z_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) \cdot \epsilon_i $$
Log-determinant for loss: `log_det = sum(log(sigma))`.

### Gated Activation
```python
def gated_activation(x):
    f, g = torch.chunk(x, 2, dim=1)
    return torch.tanh(f) * torch.sigmoid(g)
```

---

## Visualization Snippets

### Plotting Reconstructions
```python
import matplotlib.pyplot as plt

def plot_recon(original, recon):
    n = 8
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(torch.sigmoid(recon[i]).cpu().detach().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()
```

---

## Experiment Ideas

- **Easy (2/5):** Disable the Flow Prior (`use_flow=False`) and compare KL divergence to a `use_flow=True` run.
- **Medium (3/5):** Sweep `decoder_layers` from 1 to 15. Plot `KL` vs `layers`. Watch KL vanish as layers increase.
- **Advanced (5/5):** Implement **PixelRNN** (using LSTMs) as the decoder instead of PixelCNN. Compare sampling speed vs quality.

---

## Success Criteria

1.  **Non-zero KL:** `D_KL` should settle between 5 and 20 bits for MNIST.
2.  **Global Coherence:** Images generated from random $z$ must look like digits (7s, 1s, etc.), not just "furry noise."
3.  **Sharp Edges:** Reconstructions should be sharper than a standard MLP-VAE.

---

**Next:** [Day 24 — GPipe](../24_GPipe/)
