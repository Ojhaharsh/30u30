"""
Day 20: Relational RNNs - Visualization
===================================================================

Plots to understand Relational Memory:
- Attention Heatmaps: See *where* the model looks in its own memory
- Loss Comparison: RMC vs LSTM learning curves

Run `train_minimal.py --visualize` to generate these plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_attention_heatmap(attention_weights, save_path="attention_map.png"):
    """
    Visualizes the self-attention weights of the RMC over time.
    
    Args:
        attention_weights: (seq_len, num_heads, mem_slots, mem_slots)
                           We will plot the average attention over heads.
        save_path: Path to save the image.
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
        
    seq_len, num_heads, slots, _ = attention_weights.shape
    
    # Average over heads for cleaner visualization: (seq_len, slots, slots)
    avg_attention = np.mean(attention_weights, axis=1)
    
    # We want to see how the attention matrix evolves over time.
    # We'll plot a grid of heatmaps: One for each time step (or a subset).
    
    # If sequence is long, limit to first 10 steps or sample
    plot_steps = min(seq_len, 10)
    
    fig, axes = plt.subplots(1, plot_steps, figsize=(2 * plot_steps, 2.5))
    if plot_steps == 1:
        axes = [axes]
        
    for t in range(plot_steps):
        ax = axes[t]
        # Heatmap
        im = ax.imshow(avg_attention[t], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"t={t}")
        ax.axis('off')
        
    # Add colorbar
    plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    plt.suptitle("RMC Memory Self-Attention (Avg over Heads) over Time", y=0.85)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention heatmap to {save_path}")

def plot_comparison(rmc_losses, lstm_losses, save_path="comparison.png"):
    """
    Plots training loss comparison between RMC and LSTM.
    """
    plt.figure(figsize=(10, 6))
    
    # Smooth curves slightly for readability if needed, but raw is honest.
    plt.plot(rmc_losses, label='RMC (Relational)', alpha=0.8)
    plt.plot(lstm_losses, label='LSTM (Standard)', alpha=0.8)
    
    plt.title("Training Loss: RMC vs. LSTM (N-th Farthest Task)")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot to {save_path}")
