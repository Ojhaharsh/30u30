# Day 23 Solutions: Variational Lossy Autoencoder

This directory contains the reference implementations for the Day 23 exercises.

## Overview of Solutions

### 1. Masked Convolutions (`solution_01_masked_conv.py`)
- **Key Logic:** Implementation of spatial masking. 
- **Type A:** Masks the center pixel (0). Essential for the very first layer so the model can't "see" the pixel it's trying to predict.
- **Type B:** Includes the center pixel (1). Used for all subsequent layers to allow information flow from previous layers.

### 2. Gated PixelCNN Prior (`solution_02_pixelcnn_prior.py`)
- **Key Logic:** Standalone autoregressive density estimation.
- **Goal:** Verify that your MaskedConvs can actually generate image-like structures without any latent code $z$.

### 3. Reproducing Collapse (`solution_03_posterior_collapse.py`)
- **Key Logic:** Intentional failure mode.
- **Setup:** Uses a large receptive field (e.g., 10+ layers) with a standard Gaussian prior.
- **Result:** You should observe the KL Divergence term dropping to near-zero as the decoder handles all the modeling work.

### 4. IAF Flow Prior (`solution_04_flow_prior.py`)
- **Key Logic:** Inverse Autoregressive Flow (IAF).
- **Mechanism:** Uses MADE to calculate $\mu$ and $\sigma$ for each latent dimension in a single forward pass while maintaining the autoregressive constraint.

### 5. Final Assembly (`solution_05_vlae_complete.py`)
- **Key Logic:** The complete VLAE.
- **Verification:** Successfully trains on MNIST with a non-zero KL and sharp reconstructions.

## Benchmarks (MNIST)

| Metric | Expected Value (Approx) |
|--------|-------------------------|
| **Reconstruction Loss** | ~80 - 90 nats |
| **KL Divergence** | 10 - 20 bits |
| **Total ELBO** | ~100 nats |

---
**Next:** [Day 24 — GPipe](../../24_GPipe/)
