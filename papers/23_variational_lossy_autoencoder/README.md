# Day 23: Variational Lossy Autoencoder (VLAE)

> Xi Chen, Kingma, Salimans, et al. (2017) — [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731)

**Time:** 3-4 hours  
**Prerequisites:** VAE Essentials (Day 4/5), PixelCNN/RNN Intuition  
**Code:** PyTorch + IAF (Inverse Autoregressive Flow)

---

## What This Paper Is Actually About

This paper solves one of the most frustrating problems in deep generative modeling: **Posterior Collapse**. 

When you pair a Variational Autoencoder (VAE) with a powerful autoregressive decoder (like PixelCNN), the model often chooses the "lazy" path. The decoder becomes so good at predicting pixels from their immediate neighbors that it stops bothering to look at the global latent code $z$. The result? A model that ignores your input and generates generic output.

The VLAE fix is an act of **forced learning**: by "crippling" the decoder's local vision (receptive field) and boosting the prior's flexibility with **normalizing flows**, the authors force the model to encode global semantic structure into $z$ while the decoder handles the local texture.

---

## The Core Idea

The information preference of a model depends on the capacity of its components. If the decoder can explain the data locally, it will.

```
KL (Latent Usage)
    ^
    |      * PEAK (VLAE)
    |     / \
    |    /   \
    |   /     \
    |  /       \_________________ COLLAPSE (PixelCNN VAE)
    +----------------------------------> Receptive Field Size
```

VLAE operates at the **left side** of this curve. By keeping the receptive field small (e.g., 7x7 or 11x11), the model *must* use $z$ to figure out where the major object boundaries are.

---

## What the Authors Actually Showed

1.  **State-of-the-art Compression:** VLAE achieved 2.95 bits/dim on CIFAR-10, outperforming standard VAEs and competing with pure autoregressive models.
2.  **Information Decomposition:** Visually proved that $z$ captures global shape/identity while the decoder fills in consistent local patterns.
3.  **Prior Flexibility:** Demonstrated that learning the prior $p(z)$ via **Inverse Autoregressive Flow (IAF)** significantly improves the Evidence Lower Bound (ELBO).

---

## The Architecture

The VLAE consists of three "elite" components:

1.  **The Encoder $q(z|x)$:** A ResNet that compresses images into latent parameters.
2.  **The Flow Prior $p(z)$:** A sequence of IAF blocks that transforms a simple Gaussian into a complex, learnable distribution.
3.  **The Restricted Decoder $p(x|z)$:** A Gated PixelCNN with limited depth and kernel size, ensuring a small receptive field.

---

## Implementation Notes

- **Autoregressive Property:** Enforced via **MaskedConv2d** (Type A and Type B masks).
- **Gated Units:** Uses `tanh * sigmoid` gating to avoid vanishing gradients in deep PixelCNNs.
- **IAF Implementation:** Uses **MADE** (Masked Autoencoder for Distribution Estimation) to compute mu/sigma parameters in parallel while maintaining autoregression.

---

## What to Build

### Quick Start

```bash
# Verify the implementation
python setup.py

# Train the VLAE on binarized MNIST
python train_minimal.py --epochs 10 --use-flow
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | `exercise_01_masked_conv.py` | Implement the Type A/B masking logic. |
| 2 | `exercise_02_pixelcnn_prior.py` | Build a standalone density estimator. |
| 3 | `exercise_03_posterior_collapse.py` | Reproduce collapse by increasing RF size. |
| 4 | `exercise_04_flow_prior.py` | Implement the MADE-based IAF blocks. |
| 5 | `exercise_05_vlae_complete.py` | Final assembly: Cure the collapse. |

---

## Key Takeaways

1.  **Limitation is a Feature:** Restricting a model's local expressive power forces it to learn better global representations.
2.  **Inductive Bias Matters:** Masking and receptive field control are powerful tools for controlling information flow.
3.  **VAEs + Flows = Power:** Normalizing flows remove the "Gaussian bottleneck" of standard VAEs.

---

## Files in This Directory

| File | What It Is |
|------|------------|
| `implementation.py` | The main PyTorch implementation of VLAE and IAF. |
| `paper_notes.md` | Deep dive into the math and theoretical context. |
| `CHEATSHEET.md` | Quick reference, hyperparameters, and pro tips. |
| `visualization.py` | Suite for plotting KL curves and reconstructions. |
| `notebook.ipynb` | Interactive walkthrough and latent space exploration. |

---

## Further Reading

- [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) (Germaine et al., 2015)
- [Conditional Image Generation with PixelCNN](https://arxiv.org/abs/1611.02251) (van den Oord et al., 2016)
- [Improved Variational Inference with IAF](https://arxiv.org/abs/1606.04934) (Kingma et al., 2016)

---

**Previous:** [Day 22 — VQ-VAE](../22_VQ_VAE/)  
**Next:** [Day 24 — GPipe](../24_GPipe/)
