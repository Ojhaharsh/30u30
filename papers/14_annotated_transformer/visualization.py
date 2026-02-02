"""
Visualization Utilities for The Annotated Transformer
=====================================================

Provides visualization functions for:
- Attention weight heatmaps
- Positional encoding patterns
- Learning rate schedules
- Training curves
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_weights(attention, tokens_src=None, tokens_tgt=None, head=0, layer=0):
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention: Attention weights (heads, seq_q, seq_k) or (batch, heads, seq_q, seq_k)
        tokens_src: Source token labels (optional)
        tokens_tgt: Target token labels (optional)
        head: Which head to visualize
        layer: Layer index (for title)
    """
    if attention.dim() == 4:
        attention = attention[0]  # Take first batch
    
    attn = attention[head].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attn,
        xticklabels=tokens_src if tokens_src else range(attn.shape[1]),
        yticklabels=tokens_tgt if tokens_tgt else range(attn.shape[0]),
        cmap='Blues',
        annot=True if attn.shape[0] <= 10 else False,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_xlabel('Source Position (Key)')
    ax.set_ylabel('Target Position (Query)')
    ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
    
    plt.tight_layout()
    return fig


def plot_all_heads(attention, tokens_src=None, tokens_tgt=None, layer=0):
    """
    Plot attention weights for all heads in a grid.
    
    Args:
        attention: Attention weights (heads, seq_q, seq_k)
        tokens_src: Source token labels
        tokens_tgt: Target token labels
        layer: Layer index
    """
    if attention.dim() == 4:
        attention = attention[0]
    
    n_heads = attention.size(0)
    cols = 4
    rows = (n_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for head in range(n_heads):
        attn = attention[head].detach().cpu().numpy()
        ax = axes[head]
        
        sns.heatmap(
            attn,
            cmap='Blues',
            ax=ax,
            cbar=False,
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(f'Head {head}')
    
    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'All Attention Heads - Layer {layer}', fontsize=14)
    plt.tight_layout()
    return fig


def plot_positional_encoding(pe, d_model=512, max_len=100):
    """
    Visualize positional encoding patterns.
    
    Args:
        pe: PositionalEncoding module or tensor (seq, d_model)
        d_model: Model dimension
        max_len: Number of positions to show
    """
    if hasattr(pe, 'pe'):
        # It's a PositionalEncoding module
        encoding = pe.pe[0, :max_len, :].detach().cpu().numpy()
    else:
        encoding = pe[:max_len, :].detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full heatmap
    ax = axes[0, 0]
    im = ax.imshow(encoding.T, aspect='auto', cmap='RdBu')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Positional Encoding Heatmap')
    plt.colorbar(im, ax=ax)
    
    # First few dimensions
    ax = axes[0, 1]
    for dim in range(min(8, encoding.shape[1])):
        ax.plot(encoding[:, dim], label=f'dim {dim}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('First 8 Dimensions')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Sin/Cos pattern
    ax = axes[1, 0]
    ax.plot(encoding[:, 0], label='dim 0 (sin)')
    ax.plot(encoding[:, 1], label='dim 1 (cos)')
    ax.plot(encoding[:, 2], label='dim 2 (sin)')
    ax.plot(encoding[:, 3], label='dim 3 (cos)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Sin/Cos Pairs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Similarity matrix (dot product between positions)
    ax = axes[1, 1]
    similarity = np.dot(encoding, encoding.T)
    im = ax.imshow(similarity, cmap='viridis')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Position Similarity (Dot Product)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_learning_rate_schedule(d_model=512, warmup=4000, steps=20000):
    """
    Plot the Noam learning rate schedule.
    
    Args:
        d_model: Model dimension
        warmup: Warmup steps
        steps: Total steps to plot
    """
    def rate(step, d_model, warmup):
        if step == 0:
            step = 1
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    
    rates = [rate(i, d_model, warmup) for i in range(steps)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(rates)
    ax.axvline(x=warmup, color='r', linestyle='--', label=f'Warmup ({warmup} steps)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'Noam Learning Rate Schedule (d_model={d_model}, warmup={warmup})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_curves(losses, val_losses=None):
    """
    Plot training and validation loss curves.
    
    Args:
        losses: List of training losses per epoch
        val_losses: Optional list of validation losses
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(losses, label='Training Loss', marker='o')
    if val_losses:
        ax.plot(val_losses, label='Validation Loss', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_label_smoothing(smoothing_values=[0.0, 0.1, 0.2, 0.3], vocab_size=10):
    """
    Visualize the effect of label smoothing on target distributions.
    
    Args:
        smoothing_values: List of smoothing values to compare
        vocab_size: Vocabulary size
    """
    fig, axes = plt.subplots(1, len(smoothing_values), figsize=(4 * len(smoothing_values), 4))
    
    true_class = 3
    
    for ax, smooth in zip(axes, smoothing_values):
        dist = np.ones(vocab_size) * (smooth / (vocab_size - 1))
        dist[true_class] = 1.0 - smooth
        
        colors = ['steelblue'] * vocab_size
        colors[true_class] = 'coral'
        
        ax.bar(range(vocab_size), dist, color=colors)
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Probability')
        ax.set_title(f'Smoothing = {smooth}')
        ax.set_ylim(0, 1.1)
    
    fig.suptitle('Label Smoothing Effect on Target Distribution', fontsize=14)
    plt.tight_layout()
    return fig


def plot_mask(mask, title='Attention Mask'):
    """
    Visualize an attention mask.
    
    Args:
        mask: Boolean or float mask tensor
        title: Plot title
    """
    if mask.dim() > 2:
        mask = mask.squeeze()
    
    mask_np = mask.detach().cpu().numpy().astype(float)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        mask_np,
        cmap='Greens',
        annot=True if mask_np.shape[0] <= 10 else False,
        fmt='.0f',
        cbar=True,
        ax=ax
    )
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION DEMO")
    print("=" * 60)
    
    # 1. Learning rate schedule
    print("\n[1] Plotting learning rate schedule...")
    fig = plot_learning_rate_schedule()
    plt.savefig('lr_schedule.png', dpi=150)
    print("    Saved: lr_schedule.png")
    
    # 2. Positional encoding
    print("\n[2] Plotting positional encoding...")
    pe = torch.zeros(100, 512)
    position = torch.arange(0, 100).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, 512, 2).float() * -(math.log(10000.0) / 512))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    fig = plot_positional_encoding(pe)
    plt.savefig('positional_encoding.png', dpi=150)
    print("    Saved: positional_encoding.png")
    
    # 3. Label smoothing
    print("\n[3] Plotting label smoothing effect...")
    fig = plot_label_smoothing()
    plt.savefig('label_smoothing.png', dpi=150)
    print("    Saved: label_smoothing.png")
    
    # 4. Subsequent mask
    print("\n[4] Plotting subsequent mask...")
    size = 6
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
    fig = plot_mask(mask, 'Causal (Subsequent) Mask')
    plt.savefig('subsequent_mask.png', dpi=150)
    print("    Saved: subsequent_mask.png")
    
    # 5. Fake attention weights
    print("\n[5] Plotting sample attention weights...")
    attn = torch.softmax(torch.randn(8, 6, 6), dim=-1)
    fig = plot_all_heads(attn, layer=0)
    plt.savefig('attention_heads.png', dpi=150)
    print("    Saved: attention_heads.png")
    
    print("\n" + "=" * 60)
    print("All visualizations saved!")
    print("=" * 60)
    
    plt.show()
