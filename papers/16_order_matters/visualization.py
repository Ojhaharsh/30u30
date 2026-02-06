"""
Visualization Utilities for Pointer Networks
============================================

Tools for visualizing attention mechanisms and geometric problem solving.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_attention_heatmap(
    attention_weights: List[torch.Tensor],
    input_labels: List[str],
    output_labels: List[str],
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualizes attention weights as a heatmap.
    """
    attention_matrix = torch.stack(attention_weights).cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Attention Weight')
    
    ax.set_xticks(range(len(input_labels)))
    ax.set_xticklabels(input_labels)
    ax.set_yticks(range(len(output_labels)))
    ax.set_yticklabels(output_labels)
    
    ax.set_xlabel('Input Elements')
    ax.set_ylabel('Output Steps')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def visualize_sorting(
    input_values: torch.Tensor,
    pointers: torch.Tensor,
    attention_weights: List[torch.Tensor],
    save_path: Optional[str] = None
):
    input_values = input_values.cpu().numpy().squeeze()
    pointers = pointers.cpu().numpy()
    fig = plt.figure(figsize=(16, 5))
    
    # Input panel
    ax1 = plt.subplot(1, 3, 1)
    ax1.bar(np.arange(len(input_values)), input_values)
    ax1.set_title('Unsorted Input')
    
    # Attention panel
    ax2 = plt.subplot(1, 3, 2)
    matrix = torch.stack(attention_weights).cpu().numpy()
    ax2.imshow(matrix, cmap='YlOrRd')
    ax2.set_title('Attention Weights')
    
    # Output panel
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(np.arange(len(input_values)), input_values[pointers])
    ax3.set_title('Sorted Output')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded.")
