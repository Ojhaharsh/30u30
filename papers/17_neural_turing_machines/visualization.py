import torch
import matplotlib.pyplot as plt
import numpy as np

"""
NTM Visualization Tools
Goal: Visualize the differentiable addressing focus (the 'weightings') 
as the model processes a sequence, similar to Figure 4 in Graves et al. (2014).
"""

def plot_memory_weights(weights_list, title="Memory Addressing Weighting (Read Head)"):
    """
    Visualizes how the head focus moves across memory locations over time.
    X-axis: Timesteps
    Y-axis: Memory Index (N)
    """
    # Extract weights for the first batch item
    # weights_list is a list of [batch, N] tensors
    weights = torch.stack(weights_list).detach().cpu().numpy()[:, 0, :]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(weights.T, aspect='auto', cmap='hot')
    plt.xlabel("Timesteps (t)")
    plt.ylabel("Memory Location Index (i)")
    plt.title(title)
    plt.colorbar(label="Addressing Weight w_t(i)")
    plt.tight_layout()
    plt.savefig("papers/17_neural_turing_machines/addressing_focus.png")
    plt.show()

def plot_recall_performance(inputs, targets, outputs, title="Copy Task: Input vs. Recall"):
    """
    Qualitative check: how well does the bitmask recall match the target?
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Input phase
    axes[0].imshow(inputs.detach().cpu().numpy().T, aspect='auto', cmap='binary')
    axes[0].set_title("Input Sequence (with Channel 9 Delimiter)")
    
    # Target
    axes[1].imshow(targets.detach().cpu().numpy().T, aspect='auto', cmap='binary')
    axes[1].set_title("Expected Output (Target)")
    
    # Preds
    preds = (outputs > 0).float().detach().cpu().numpy()
    axes[2].imshow(preds.T, aspect='auto', cmap='binary')
    axes[2].set_title("Actual Model Recall (Predictions)")
    
    for ax in axes:
        ax.set_ylabel("Bit Index")
    
    plt.tight_layout()
    plt.savefig("papers/17_neural_turing_machines/recall_comparison.png")
    plt.show()
