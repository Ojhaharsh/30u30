"""
Visualization Tools for Pointer Networks
Standardized for Day 18 of 30u30

Functions to visualize:
1. Attention Heatmaps (Pointer distributions)
2. Sorting results verification
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_attention(input_values, attention_probs, save_path="attention_viz.png"):
    """
    Plot the pointer probability matrix.
    
    Args:
        input_values (list): The original input values (numbers, text, etc.)
        attention_probs (ndarray): (seq_len, seq_len) matrix of probabilities.
        save_path (str): File path to save the generated image.
    """
    seq_len = len(input_values)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(attention_probs, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, label='Pointer Strength')
    
    # Axis labeling
    # X-axis: The input items the model can point to
    plt.xticks(range(seq_len), [f"{v:.2f}" for v in input_values])
    # Y-axis: The output steps (what the model picked first, second, etc.)
    plt.yticks(range(seq_len), [f"Output Step {i+1}" for i in range(seq_len)])
    
    plt.title("Pointer Network: Attention Heatmap", fontsize=14, fontweight='bold')
    plt.xlabel("Input Sequence Elements", fontsize=12)
    plt.ylabel("Decoding Time Step", fontsize=12)
    
    # Grid lines for clarity
    plt.vlines(np.arange(-0.5, seq_len), -0.5, seq_len-0.5, colors='grey', alpha=0.1)
    plt.hlines(np.arange(-0.5, seq_len), -0.5, seq_len-0.5, colors='grey', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Visualization saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Pointer Network Visualization Tools")
    print("=" * 60)
    
    # Dummy data for demonstration
    # Sequence: 0.7, 0.2, 0.9, 0.1, 0.5
    # Sorted: 0.1 (idx 3), 0.2 (idx 1), 0.5 (idx 4), 0.7 (idx 0), 0.9 (idx 2)
    inputs = [0.7, 0.2, 0.9, 0.1, 0.5]
    
    # Create a sharp "trained" looking matrix
    dummy_attn = np.zeros((5, 5))
    dummy_attn[0, 3] = 0.9  # Pick 0.1
    dummy_attn[1, 1] = 0.95 # Pick 0.2
    dummy_attn[2, 4] = 0.92 # Pick 0.5
    dummy_attn[3, 0] = 0.98 # Pick 0.7
    dummy_attn[4, 2] = 0.99 # Pick 0.9
    
    # Add noise to simulate real model behavior
    dummy_attn += np.random.rand(5, 5) * 0.05
    # Renormalize rows to sum to 1
    dummy_attn /= dummy_attn.sum(axis=1, keepdims=True)
    
    plot_attention(inputs, dummy_attn)
