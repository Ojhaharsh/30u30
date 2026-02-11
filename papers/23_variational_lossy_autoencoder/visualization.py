"""
visualization.py - Visualization Suite for Variational Lossy Autoencoder (VLAE)

Generates key plots demonstrating the core themes of Chen et al. (2017):
1. Posterior Convergence: KL Divergence vs. Reconstruction Loss over time.
2. Global vs Local: Side-by-side comparison of reconstructions.
3. Latent Sampling: Autoregressive pixel-by-pixel generation.
4. KL Heatmap: Spatial distribution of information bits.
5. Latent Traversals: Demonstrating high-level semantic capture in z.

Reference: Chen et al. (2017) - https://arxiv.org/abs/1611.02731
Author: 30u30 Project
"""

import argparse
import sys
import os
import torch
import numpy as np
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib is required for visualization.")
    print("Install with: pip install matplotlib")
    sys.exit(1)

from implementation import VLAE


def plot_training_trajectories(history: Dict[str, List[float]], output_dir: str = '.'):
    """
    Plot 1: Training convergence. 
    Crucial for VLAE to show that KL stays non-zero (No Collapse).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reconstruction Loss (BCE)', color=color)
    ax1.plot(history['train_recon'], color=color, linewidth=2, label='Recon Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('KL Divergence (Bits)', color=color)
    ax2.plot(history['train_kl'], color=color, linewidth=2, label='KL Divergence')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("VLAE Training Trajectory: Monitoring Posterior Collapse", fontsize=14)
    fig.tight_layout()
    
    outpath = os.path.join(output_dir, "plot1_training_trajectory.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_reconstructions(model, data, device, output_dir='.'):
    """
    Plot 2: Original vs Reconstruction.
    Visual assessment of the 'lossy' vs 'textured' trade-off.
    """
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = model(data.to(device))
        recon = torch.sigmoid(logits).cpu()

    n = 8
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        # Original
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original", loc='left')
        
        # Recon
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("VLAE Recon", loc='left')

    plt.suptitle("VLAE Reconstructions (Local Texture + Global Structure)", fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "plot2_reconstructions.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_latent_sampling(model, device, latent_dim, n_samples=16, output_dir='.'):
    """
    Plot 3: True autoregressive sampling from the latent prior.
    This is slow because it happens pixel-by-pixel.
    """
    print("  Generating autoregressive samples (pixel-by-pixel)...")
    model.eval()
    with torch.no_grad():
        # Sample random z from prior (Standard normal for simplicity in demo)
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # Start with empty image
        samples = torch.zeros(n_samples, 1, 28, 28).to(device)
        
        # Spatially expand z
        h_latent = z.view(n_samples, latent_dim, 1, 1).expand(-1, -1, 28, 28)

        # Autoregressive loop
        for i in range(28):
            for j in range(28):
                # We reuse the decoder path
                # Feed the current 'samples' (partially filled) back to model
                h = model.initial_conv(samples)
                for layer in model.decoder_layers:
                    h = layer(h, h_latent)
                logits = model.final_conv(h)
                
                # Sample the current pixel
                probs = torch.sigmoid(logits[:, :, i, j])
                samples[:, :, i, j] = torch.bernoulli(probs)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(n_samples):
        r, c = i // 4, i % 4
        axes[r, c].imshow(samples[i].cpu().squeeze(), cmap='gray')
        axes[r, c].axis('off')

    plt.suptitle("Samples from Prior $p(z)$ via Autoregressive Decoding", fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "plot3_sampling.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_bits_heatmap(mu, logvar, output_dir='.'):
    """
    Plot 4: Visualize how many 'bits' are used in each latent dimension.
    Based on KL = 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
    """
    kl_per_dim = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    avg_kl = kl_per_dim.mean(dim=0).cpu().detach().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(avg_kl)), avg_kl, color='skyblue', edgecolor='navy')
    plt.xlabel("Latent Dimension")
    plt.ylabel("KL (Bits)")
    plt.title("Information Content per Latent Dimension (Bit Allocation)")
    plt.grid(axis='y', alpha=0.3)
    
    outpath = os.path.join(output_dir, "plot4_bits_per_dim.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_latent_traversals(model, data, device, latent_dim, dim_idx=0, output_dir='.'):
    """
    Plot 5: Latent space traversals.
    Sweeps a latent dimension to show semantic control.
    """
    model.eval()
    with torch.no_grad():
        # Encode a single sample
        mu, logvar = model.encoder(data[0:1].to(device))
        z_base = mu # Use mu for a clean baseline
        
        n_steps = 8
        traversals = []
        # Move z in the range [-3, 3]
        for val in np.linspace(-3, 3, n_steps):
            z_step = z_base.clone()
            z_step[0, dim_idx] = val
            
            # Reconstruction (Note: Using fast non-autoregressive recon for visualization)
            h_latent = z_step.view(1, -1, 1, 1).expand(-1, -1, 28, 28)
            h = model.initial_conv(torch.zeros(1, 1, 28, 28).to(device))
            for layer in model.decoder_layers:
                h = layer(h, h_latent)
            logits = model.final_conv(h)
            traversals.append(torch.sigmoid(logits).cpu().squeeze())

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2.5))
    for i, img in enumerate(traversals):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"z[{dim_idx}]={np.linspace(-3,3,n_steps)[i]:.1f}", fontsize=8)

    plt.suptitle(f"Latent Traversal: Dimension {dim_idx}", fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"plot5_traversal_dim{dim_idx}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLAE Visualization Suite")
    parser.add_argument('--model-path', type=str, default='vlae_mnist.pth')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=" * 60)
    print("VLAE Visualization Suite (Chen et al. 2017)")
    print("=" * 60)
    
    # In a real scenario, we'd load the model and data here.
    # For this suite, we provide the standalone functions to be called 
    # from train_minimal.py or the notebook.
    print("Utility script loaded. Call functions with trained model/data.")
