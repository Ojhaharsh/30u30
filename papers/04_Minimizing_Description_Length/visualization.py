"""
Visualization Tools for MDL / Bayesian Networks
===============================================

Comprehensive suite for analyzing "Noisy Weight" Networks.

This file provides deep introspection into the Bayesian Brain:
1. Uncertainty Envelopes (The visual proof of doubt)
2. Weight Distributions (The "Bits Back" compression visual)
3. Signal-to-Noise Analysis (Which neurons actually matter?)
4. Prior-Posterior Shifts (How much did the model learn?)
5. Complexity Dynamics (The battle between Accuracy and Simplicity)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

# Set style for professional-grade plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_uncertainty_envelope(model, X_train, y_train, X_test=None, n_samples=100, save_path=None):
    """
    Visualize the "Confidence" of the model (The Holy Grail of Bayesian NN).
    
    This plot demonstrates the core value of MDL:
    - Near data: Model is confident (low variance, narrow tube).
    - Far from data: Model is uncertain (high variance, wide tube).
    
    Args:
        model: Trained MDLNetwork instance
        X_train, y_train: Training data points
        X_test: Test range (default: auto-detected)
        n_samples: Number of Monte Carlo forward passes
    """
    print(f"  [Visualizer] Generating Uncertainty Envelope ({n_samples} samples)...")
    
    if X_test is None:
        # Auto-range: Go 50% wider than training data
        span = X_train.max() - X_train.min()
        X_test = np.linspace(X_train.min() - span*0.5, 
                             X_train.max() + span*0.5, 
                             300).reshape(-1, 1)
    
    # Monte Carlo Sampling
    preds = []
    for _ in range(n_samples):
        # Every forward pass uses different random noise!
        preds.append(model.forward(X_test))
    
    preds = np.array(preds).squeeze() # (n_samples, n_points)
    
    # Statistics
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. Spaghetti Plot (The Ensemble)
    # Plot faint lines to show individual "hypotheses"
    for i in range(min(20, n_samples)):
        ax.plot(X_test, preds[i], color='teal', alpha=0.15, linewidth=1)
        
    # 2. Mean Prediction
    ax.plot(X_test, mu, color='navy', linewidth=3, label='Mean Prediction')
    
    # 3. Uncertainty Bands (Confidence Intervals)
    # 1 Sigma (68%)
    ax.fill_between(X_test.flatten(), mu - sigma, mu + sigma, 
                    color='teal', alpha=0.3, label='68% Confidence ($\pm 1\sigma$)')
    # 2 Sigma (95%)
    ax.fill_between(X_test.flatten(), mu - 2*sigma, mu + 2*sigma, 
                    color='teal', alpha=0.1, label='95% Confidence ($\pm 2\sigma$)')
    
    # 4. Training Data
    ax.scatter(X_train, y_train, color='crimson', s=50, edgecolors='white', 
               linewidth=1.5, zorder=10, label='Training Data')
    
    # Formatting
    ax.set_title("The Bayesian Uncertainty Envelope", fontsize=18, fontweight='bold')
    ax.set_xlabel('Input Space ($x$)', fontsize=14)
    ax.set_ylabel('Model Output ($y$)', fontsize=14)
    ax.legend(frameon=True, fontsize=12, loc='upper left')
    
    # Annotations for educational value
    low_uncertainty_idx = np.argmin(sigma)
    high_uncertainty_idx = np.argmax(sigma)
    
    ax.annotate('Low Uncertainty\n(Model "Knows")', 
                xy=(X_test[low_uncertainty_idx], mu[low_uncertainty_idx]), 
                xytext=(X_test[low_uncertainty_idx], mu[low_uncertainty_idx]-1.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
                
    ax.annotate('High Uncertainty\n(Model "Guesses")', 
                xy=(X_test[high_uncertainty_idx], mu[high_uncertainty_idx] + 2*sigma[high_uncertainty_idx]), 
                xytext=(X_test[high_uncertainty_idx]-1, mu[high_uncertainty_idx] + 3),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [Saved] {save_path}")
    else:
        plt.show()

def plot_weight_distributions(model, layer_name='layer1', save_path=None):
    """
    Visualize the Anatomy of Belief (Mu vs Sigma).
    
    This visualizes the "Description Length" directly.
    - Sharp peaks (low sigma) = High Information cost.
    - Flat hills (high sigma) = Low Information cost.
    """
    print(f"  [Visualizer] Analyzing Weight Distributions for {layer_name}...")
    layer = getattr(model, layer_name)
    
    # Extract parameters
    mu = layer.w_mu.flatten()
    # Calculate sigma from rho (Softplus)
    sigma = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
    sigma = sigma.flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Histogram of Means (What values?)
    sns.histplot(mu, kde=True, ax=axes[0], color='indigo', bins=30)
    axes[0].set_title(f'Weight Means ($\mu$)\n"The Best Guesses"', fontweight='bold')
    axes[0].set_xlabel('Weight Value')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Histogram of Uncertainties (How sure?)
    sns.histplot(sigma, kde=True, ax=axes[1], color='darkorange', bins=30)
    axes[1].set_title(f'Weight Uncertainty ($\sigma$)\n"The Fuzziness"', fontweight='bold')
    axes[1].set_xlabel('Standard Deviation')
    
    # Plot 3: The Volcano Plot (Significance)
    # Large weights usually need low uncertainty to be useful.
    # Small weights can have high uncertainty (they don't matter).
    sns.scatterplot(x=mu, y=sigma, ax=axes[2], alpha=0.6, color='seagreen')
    axes[2].set_title('The Belief Landscape ($\mu$ vs $\sigma$)', fontweight='bold')
    axes[2].set_xlabel('Mean Weight Value ($\mu$)')
    axes[2].set_ylabel('Uncertainty ($\sigma$)')
    axes[2].axvline(0, color='black', linestyle='--', alpha=0.3)
    
    # Add "Pruning Zone" annotation
    axes[2].add_patch(plt.Rectangle((-0.5, 0.2), 1.0, 1.0, 
                                    color='red', alpha=0.1, label='Prunable Zone'))
    axes[2].text(0, 0.5, "High Noise,\nLow Value\n(Prunable)", 
                 ha='center', va='center', color='darkred', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [Saved] {save_path}")
    else:
        plt.show()

def plot_snr_analysis(model, threshold=1.0, save_path=None):
    """
    Signal-to-Noise Ratio (SNR) Analysis.
    
    SNR = |mu| / sigma
    
    This is the ultimate test of "Did the network learn this weight?"
    - High SNR (>1): The weight is definitely positive or negative. Important.
    - Low SNR (<1): The noise is larger than the value. The weight is effectively random.
    """
    print("  [Visualizer] Calculating Signal-to-Noise Ratios...")
    
    layers = [('Layer 1', model.layer1), ('Layer 2', model.layer2)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (name, layer) in zip(axes, layers):
        mu = layer.w_mu.flatten()
        sigma = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
        sigma = sigma.flatten()
        
        # Calculate SNR
        snr = np.abs(mu) / (sigma + 1e-9)
        
        # Sort for visualization
        sorted_snr = np.sort(snr)[::-1]
        
        # Plot
        ax.plot(sorted_snr, linewidth=2, color='crimson')
        ax.axhline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
        
        # Shading
        ax.fill_between(range(len(snr)), 0, sorted_snr, where=(sorted_snr > threshold),
                        color='green', alpha=0.3, label='Effective Weights')
        ax.fill_between(range(len(snr)), 0, sorted_snr, where=(sorted_snr <= threshold),
                        color='gray', alpha=0.3, label='Noise Weights')
        
        # Stats
        effective_count = np.sum(snr > threshold)
        total_count = len(snr)
        
        ax.set_title(f'{name} SNR Spectrum\n({effective_count}/{total_count} Effective)', 
                     fontweight='bold')
        ax.set_xlabel('Weight Index (Sorted)')
        ax.set_ylabel('Signal-to-Noise Ratio (|$\mu$| / $\sigma$)')
        ax.set_yscale('log')
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [Saved] {save_path}")
    else:
        plt.show()

def plot_prior_posterior_shift(model, save_path=None):
    """
    Visualize Learning as "Moving away from the Prior".
    
    We start with a Prior: N(0, 1) (or similar).
    We end with a Posterior: N(mu, sigma).
    The difference is the KNOWLEDGE gained.
    """
    print("  [Visualizer] Plotting Prior vs Posterior Shift...")
    
    # Collect all weights from all layers
    all_mu = []
    all_sigma = []
    
    for layer in [model.layer1, model.layer2]:
        all_mu.append(layer.w_mu.flatten())
        s = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
        all_sigma.append(s.flatten())
        
    all_mu = np.concatenate(all_mu)
    all_sigma = np.concatenate(all_sigma)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Plot Prior (Standard Normal)
    x_axis = np.linspace(-4, 4, 1000)
    ax.plot(x_axis, norm.pdf(x_axis, 0, 1), 'k--', linewidth=2, label='Prior Belief $N(0,1)$')
    
    # 2. Plot Posterior Aggregate
    # We approximate the posterior density by KDE of the samples
    sns.kdeplot(all_mu, fill=True, color='blue', alpha=0.3, 
                label='Posterior Means (Learned)', ax=ax)
    
    ax.set_title("How much did the model learn?\n(Divergence from Prior)", fontsize=16, fontweight='bold')
    ax.set_xlabel('Weight Space')
    ax.set_ylabel('Density')
    ax.legend(fontsize=12)
    
    # Add interpretation text
    ax.text(2.5, 0.3, "Peaks away from 0\nindicate strong features", 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [Saved] {save_path}")
    else:
        plt.show()

def plot_loss_dynamics(history, save_path=None):
    """
    The Battle: Complexity vs Error.
    Visualizes the optimization trajectory.
    """
    print("  [Visualizer] Plotting Loss Dynamics...")
    epochs = range(len(history['total']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Main Losses
    ax1.plot(epochs, history['total'], 'k-', linewidth=2, label='Total Loss (Objective)')
    ax1.plot(epochs, history['nll'], 'b--', linewidth=2, label='Error (NLL)')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Training Trajectory: Minimizing Free Energy', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Complexity Cost (KL)
    # This often GOES UP as the model learns structure, then stabilizes
    ax2.plot(epochs, history['kl'], 'r-', linewidth=2, label='Complexity Cost (KL)')
    ax2.set_ylabel('Information Cost (Nats)')
    ax2.set_xlabel('Epochs')
    ax2.set_title('Description Length Cost (The "File Size" of the Model)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [Saved] {save_path}")
    else:
        plt.show()

def analyze_compression_stats(model, threshold_snr=0.5):
    """
    Generate a text report on compression efficiency.
    Calculates effective parameter count based on SNR.
    """
    print("\n" + "="*60)
    print("MDL COMPRESSION REPORT")
    print("="*60)
    
    total_params = 0
    effective_params = 0
    total_bits_theor = 0
    
    for i, layer in enumerate([model.layer1, model.layer2]):
        name = f"Layer {i+1}"
        mu = layer.w_mu.flatten()
        sigma = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
        sigma = sigma.flatten()
        
        snr = np.abs(mu) / (sigma + 1e-9)
        n_layer = len(mu)
        n_eff = np.sum(snr > threshold_snr)
        
        # Approximate Bits needed: H(w) ≈ -log2(sigma) + const
        # If sigma is large (1.0), bits ≈ 0. If sigma small (0.01), bits ≈ 7.
        bits = np.sum(np.maximum(0, -np.log2(sigma + 1e-9)))
        
        print(f"\n{name}:")
        print(f"  Total Weights:      {n_layer}")
        print(f"  Effective Weights:  {n_eff} (SNR > {threshold_snr})")
        print(f"  Sparsity Ratio:     {100*(1 - n_eff/n_layer):.1f}%")
        print(f"  Avg Uncertainty:    {np.mean(sigma):.4f}")
        
        total_params += n_layer
        effective_params += n_eff
        total_bits_theor += bits
        
    print("-" * 60)
    print(f"OVERALL SYSTEM:")
    print(f"  Original Size:    {total_params * 32} bits (Standard float32)")
    print(f"  MDL Encoded Size: {int(total_bits_theor)} bits (Theoretical)")
    print(f"  Compression Rate: {total_params * 32 / (total_bits_theor + 1):.1f}x")
    print("=" * 60 + "\n")

def create_comprehensive_report(model, history, X, y, save_dir='mdl_analysis'):
    """
    Orchestrate the full analysis suite.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting Comprehensive MDL Analysis -> {save_dir}/")
    
    # 1. Uncertainty
    plot_uncertainty_envelope(model, X, y, save_path=f"{save_dir}/1_uncertainty.png")
    
    # 2. Weights
    plot_weight_distributions(model, 'layer1', save_path=f"{save_dir}/2_weights_layer1.png")
    
    # 3. SNR
    plot_snr_analysis(model, save_path=f"{save_dir}/3_snr_analysis.png")
    
    # 4. Prior Shift
    plot_prior_posterior_shift(model, save_path=f"{save_dir}/4_prior_shift.png")
    
    # 5. Dynamics
    plot_loss_dynamics(history, save_path=f"{save_dir}/5_training_dynamics.png")
    
    # 6. Text Report
    with open(f"{save_dir}/compression_report.txt", 'w') as f:
        import sys
        original = sys.stdout
        sys.stdout = f
        analyze_compression_stats(model)
        sys.stdout = original
        
    print("Analysis Complete. All artifacts saved.")

if __name__ == "__main__":
    print("MDL Visualization Library Loaded.")
    print("Run inside 'train_mdl.py' or import in notebook.")