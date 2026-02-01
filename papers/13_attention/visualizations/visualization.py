"""
Day 13: Transformer Visualization Module

Visualizations to understand attention and transformer components.
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization functions will be limited.")


# =============================================================================
# ATTENTION VISUALIZATION
# =============================================================================

def plot_attention_weights(
    attention_weights: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
):
    """
    Plot attention weight matrix as heatmap.
    
    Args:
        attention_weights: (seq_len_q, seq_len_k) attention matrix
        x_labels: Labels for keys (columns)
        y_labels: Labels for queries (rows)
        title: Plot title
        save_path: Optional path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    seq_q, seq_k = attention_weights.shape
    
    if x_labels is None:
        x_labels = [str(i) for i in range(seq_k)]
    if y_labels is None:
        y_labels = [str(i) for i in range(seq_q)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(seq_k))
    ax.set_yticks(range(seq_q))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Keys', fontsize=12)
    ax.set_ylabel('Queries', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Add value annotations
    for i in range(seq_q):
        for j in range(seq_k):
            val = attention_weights[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_multi_head_attention(
    attention_weights: np.ndarray,
    head_names: Optional[List[str]] = None,
    title: str = "Multi-Head Attention",
    save_path: Optional[str] = None
):
    """
    Plot attention weights for multiple heads.
    
    Args:
        attention_weights: (n_heads, seq_q, seq_k) attention tensor
        head_names: Names for each head
        title: Plot title
        save_path: Optional path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    n_heads = attention_weights.shape[0]
    
    if head_names is None:
        head_names = [f"Head {i}" for i in range(n_heads)]
    
    # Determine grid size
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    
    for idx in range(n_heads):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        im = ax.imshow(attention_weights[idx], cmap='Blues', aspect='auto')
        ax.set_title(head_names[idx], fontsize=11)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for idx in range(n_heads, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# POSITIONAL ENCODING VISUALIZATION
# =============================================================================

def plot_positional_encoding(
    pe: np.ndarray,
    max_positions: int = 100,
    dims_to_show: int = 64,
    save_path: Optional[str] = None
):
    """
    Visualize positional encoding matrix.
    
    Args:
        pe: Positional encoding matrix (max_len, d_model)
        max_positions: Number of positions to show
        dims_to_show: Number of dimensions to show
        save_path: Optional path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    pe_subset = pe[:max_positions, :dims_to_show]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    ax = axes[0]
    im = ax.imshow(pe_subset.T, cmap='RdBu', aspect='auto',
                   vmin=-1, vmax=1)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    ax.set_title('Positional Encoding Heatmap', fontsize=14, weight='bold')
    plt.colorbar(im, ax=ax)
    
    # Sine waves for first few dimensions
    ax = axes[1]
    positions = np.arange(max_positions)
    
    for dim in range(0, min(8, dims_to_show), 2):
        ax.plot(positions, pe_subset[:, dim], 
               label=f'dim {dim} (sin)', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Encoding Value', fontsize=12)
    ax.set_title('Positional Encoding Curves', fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_position_similarity(
    pe: np.ndarray,
    positions: List[int] = [0, 10, 20, 50],
    save_path: Optional[str] = None
):
    """
    Show similarity between positional encodings at different positions.
    
    The key insight: positions close together have similar encodings.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    n_positions = pe.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for ref_pos in positions:
        # Compute dot product similarity with all positions
        ref_encoding = pe[ref_pos]
        similarities = np.dot(pe[:min(100, n_positions)], ref_encoding)
        
        ax.plot(range(len(similarities)), similarities, 
               label=f'Reference pos {ref_pos}', linewidth=2)
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Similarity (dot product)', fontsize=12)
    ax.set_title('Position Similarity in Sinusoidal Encoding', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# TRANSFORMER ARCHITECTURE VISUALIZATION
# =============================================================================

def plot_transformer_architecture(save_path: Optional[str] = None):
    """
    Draw a schematic of the Transformer architecture.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Helper function to draw boxes
    def draw_box(x, y, w, h, text, color='lightblue', fontsize=9):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=fontsize, weight='bold')
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Title
    ax.text(6, 9.5, 'The Transformer Architecture', ha='center', 
           fontsize=16, weight='bold')
    
    # Encoder side
    ax.text(3, 8.8, 'ENCODER', ha='center', fontsize=12, weight='bold', color='blue')
    
    # Encoder input
    draw_box(2, 7.8, 2, 0.5, 'Input Embedding', 'lightyellow')
    draw_arrow(3, 7.8, 3, 7.3)
    draw_box(2, 6.8, 2, 0.5, '+ Positional Enc', 'lightyellow')
    draw_arrow(3, 6.8, 3, 6.3)
    
    # Encoder block
    draw_box(1.5, 3.5, 3, 2.8, '', 'lightblue')
    ax.text(3, 6.1, 'Encoder Block x N', ha='center', fontsize=10, style='italic')
    draw_box(2, 5.3, 2, 0.5, 'Multi-Head Attn', 'lightskyblue')
    draw_box(2, 4.6, 2, 0.5, 'Add & Norm', 'lightgreen')
    draw_box(2, 3.9, 2, 0.5, 'Feed Forward', 'lightskyblue')
    draw_box(2, 3.2, 2, 0.5, 'Add & Norm', 'lightgreen')
    
    # Decoder side
    ax.text(9, 8.8, 'DECODER', ha='center', fontsize=12, weight='bold', color='red')
    
    # Decoder input
    draw_box(8, 7.8, 2, 0.5, 'Output Embedding', 'lightyellow')
    draw_arrow(9, 7.8, 9, 7.3)
    draw_box(8, 6.8, 2, 0.5, '+ Positional Enc', 'lightyellow')
    draw_arrow(9, 6.8, 9, 6.3)
    
    # Decoder block
    draw_box(7.5, 2.3, 3, 4, '', 'lightcoral')
    ax.text(9, 6.1, 'Decoder Block x N', ha='center', fontsize=10, style='italic')
    draw_box(8, 5.3, 2, 0.5, 'Masked Self-Attn', 'lightsalmon')
    draw_box(8, 4.6, 2, 0.5, 'Add & Norm', 'lightgreen')
    draw_box(8, 3.9, 2, 0.5, 'Cross-Attention', 'lightsalmon')
    draw_box(8, 3.2, 2, 0.5, 'Add & Norm', 'lightgreen')
    draw_box(8, 2.5, 2, 0.5, 'Feed Forward', 'lightsalmon')
    draw_box(8, 1.8, 2, 0.5, 'Add & Norm', 'lightgreen')
    
    # Cross-attention connection
    draw_arrow(4.5, 3.7, 8, 3.9)
    ax.text(6.25, 4.1, 'K, V from encoder', fontsize=8, ha='center')
    
    # Output
    draw_arrow(9, 1.8, 9, 1.3)
    draw_box(8, 0.8, 2, 0.5, 'Linear + Softmax', 'plum')
    draw_arrow(9, 0.8, 9, 0.3)
    ax.text(9, 0.1, 'Output Probabilities', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# SCALED DOT-PRODUCT VISUALIZATION
# =============================================================================

def visualize_attention_scaling(d_k_values: List[int] = [16, 64, 256, 1024],
                                save_path: Optional[str] = None):
    """
    Show why scaling by sqrt(d_k) is necessary.
    
    Large d_k → large dot products → sharp softmax → bad gradients
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Without scaling
    ax = axes[0]
    for d_k in d_k_values:
        # Random unit vectors
        q = np.random.randn(1000, d_k) / np.sqrt(d_k)
        k = np.random.randn(1000, d_k) / np.sqrt(d_k)
        
        # Dot products (without scaling)
        dots = (q * k).sum(axis=1) * d_k  # Undo normalization to show variance growth
        
        ax.hist(dots, bins=50, alpha=0.5, label=f'd_k={d_k}', density=True)
    
    ax.set_xlabel('Dot Product Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Without Scaling: Variance Grows with d_k', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # With scaling
    ax = axes[1]
    for d_k in d_k_values:
        q = np.random.randn(1000, d_k) / np.sqrt(d_k)
        k = np.random.randn(1000, d_k) / np.sqrt(d_k)
        
        # Scaled dot products
        dots = (q * k).sum(axis=1) * d_k / np.sqrt(d_k)
        
        ax.hist(dots, bins=50, alpha=0.5, label=f'd_k={d_k}', density=True)
    
    ax.set_xlabel('Scaled Dot Product Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('With Scaling: Variance is Stable', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# DEMO
# =============================================================================

def run_visualizations():
    """Run all visualization demos."""
    
    print("=" * 60)
    print("TRANSFORMER VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Random attention weights
    print("\n1. Plotting sample attention weights...")
    attn_weights = np.random.rand(6, 6)
    attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
    
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    plot_attention_weights(attn_weights, x_labels=tokens, y_labels=tokens,
                          title='Sample Self-Attention')
    
    # 2. Multi-head attention
    print("\n2. Plotting multi-head attention...")
    multi_head_attn = np.random.rand(8, 6, 6)
    multi_head_attn = multi_head_attn / multi_head_attn.sum(axis=-1, keepdims=True)
    plot_multi_head_attention(multi_head_attn)
    
    # 3. Positional encoding
    print("\n3. Plotting positional encoding...")
    from implementation import PositionalEncoding
    pe = PositionalEncoding(d_model=128)
    plot_positional_encoding(pe.pe)
    
    # 4. Position similarity
    print("\n4. Plotting position similarity...")
    plot_position_similarity(pe.pe)
    
    # 5. Scaling visualization
    print("\n5. Plotting attention scaling effect...")
    visualize_attention_scaling()
    
    # 6. Architecture diagram
    print("\n6. Plotting Transformer architecture...")
    plot_transformer_architecture()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_visualizations()
