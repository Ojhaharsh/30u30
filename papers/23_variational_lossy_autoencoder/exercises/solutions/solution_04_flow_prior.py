"""
Solution 04: IAF Flow Prior

Implementation of Inverse Autoregressive Flow (IAF) using MADE.
Used to transform a simple Gaussian into a high-capacity latent prior.
"""

import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import IAFBlock

def test_iaf_flow():
    latent_dim = 2
    # Two steps of IAF
    flow1 = IAFBlock(latent_dim, hidden_dim=64)
    flow2 = IAFBlock(latent_dim, hidden_dim=64)
    
    # Start with standard Gaussian base
    z0 = torch.randn(1000, latent_dim)
    
    # Pass through flow
    with torch.no_grad():
        z1, _ = flow1(z0)
        z2, _ = flow2(z1)
    
    # Visualize transformation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    z0_np = z0.numpy()
    axes[0].scatter(z0_np[:, 0], z0_np[:, 1], alpha=0.5, s=2)
    axes[0].set_title("Base Gaussian $z_0$")
    
    z1_np = z1.numpy()
    axes[1].scatter(z1_np[:, 0], z1_np[:, 1], alpha=0.5, s=2, color='orange')
    axes[1].set_title("After 1 IAF Step $z_1$")
    
    z2_np = z2.numpy()
    axes[2].scatter(z2_np[:, 0], z2_np[:, 1], alpha=0.5, s=2, color='red')
    axes[2].set_title("After 2 IAF Steps $z_2$")
    
    for ax in axes:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
    plt.suptitle("IAF Prior Transformation Flow", fontsize=14)
    plt.show()

if __name__ == "__main__":
    test_iaf_flow()
