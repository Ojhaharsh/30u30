"""
🔦 Visualization utilities for Bahdanau Attention

This file lets you SEE what the attention mechanism is "looking at"!

Think of these visualizations like X-ray glasses for your model:
- Attention heatmaps show WHERE the model focuses for each output word
- Evolution plots show HOW attention shifts as translation progresses
- Alignment plots reveal WHAT the model learned about language structure

Fun fact: These visualizations are how researchers discovered that attention
naturally learns grammar! "cat" → "chat", "black" → "noir" without being told!

Author: 30u30 AI Papers Project
"""

import torch
import numpy as np
from pathlib import Path

# ==============================================================================
# MATPLOTLIB IMPORT
# ==============================================================================
# We need matplotlib for all the pretty pictures!
# If you don't have it, the code will still run but you'll miss out on the fun.

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed. Install with: pip install matplotlib")
    print("   (You're missing out on the coolest part - seeing attention in action!)")


# ==============================================================================
# 🎨 ATTENTION HEATMAP
# ==============================================================================
# This is the "classic" attention visualization you see in every NMT paper.
# It's like a spreadsheet where:
#   - Rows = output words (what we're generating)
#   - Columns = input words (what we're looking at)
#   - Cell color = how much attention that output pays to that input
#
# Bright = "I'm really focusing here!"
# Dark = "Not interested in this word right now"

def plot_attention_heatmap(attention_weights, source_tokens, target_tokens,
                           title="🔍 Attention Weights Heatmap", save_path=None, 
                           figsize=(10, 8), cmap='Blues', show_values=True):
    """
    Create a beautiful attention heatmap.
    
    This is the visualization from the original Bahdanau paper!
    Each cell shows how much the decoder "looked at" each encoder position.
    
    Example output:
    
                 The    cat    sat    on     mat
        Le      [0.7]  [0.1]  [0.1]  [0.05] [0.05]  ← "Le" focuses on "The"
        chat    [0.1]  [0.8]  [0.05] [0.02] [0.03]  ← "chat" focuses on "cat"
        ...
    
    Args:
        attention_weights: 2D array (target_len, source_len)
        source_tokens: List of source tokens (displayed on x-axis)
        target_tokens: List of target tokens (displayed on y-axis)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
        cmap: Colormap name ('Blues', 'Reds', 'viridis', 'plasma')
        show_values: Whether to show attention values in cells
    
    Returns:
        matplotlib Figure object (or None if matplotlib not installed)
    """
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib required for visualization!")
        print("   Run: pip install matplotlib")
        return None
    
    # Convert to numpy if needed (PyTorch tensors → numpy arrays)
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap - this is where the magic happens!
    im = ax.imshow(attention_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar (the legend on the side)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (0=ignore, 1=focus)', fontsize=12)
    
    # Set ticks - label our rows and columns
    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, fontsize=11, rotation=45, ha='right')
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens, fontsize=11)
    
    # Labels
    ax.set_xlabel('📖 Source Sequence (what we read)', fontsize=13)
    ax.set_ylabel('✏️ Target Sequence (what we write)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Add grid - makes cells easier to read
    ax.set_xticks(np.arange(-0.5, len(source_tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_tokens), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add values in cells - so you can see exact numbers
    if show_values:
        for i in range(len(target_tokens)):
            for j in range(len(source_tokens)):
                value = attention_weights[i, j]
                # Use white text on dark cells, black on light cells
                color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    
    return fig


# ==============================================================================
# 📈 ATTENTION EVOLUTION
# ==============================================================================
# This shows how the attention "spotlight" moves as we generate each word.
# It's like watching a movie of where the model looks over time!

def plot_attention_evolution(attentions, source_tokens, target_tokens=None,
                             title="📊 Attention Evolution (Watch the Spotlight Move!)", 
                             save_path=None):
    """
    Show how attention changes at each decoding step.
    
    This is like a movie of the translation process:
    - Frame 1: Generating first word, spotlight on relevant input
    - Frame 2: Generating second word, spotlight moves
    - etc.
    
    Args:
        attentions: 2D array (target_len, source_len) or list of 1D arrays
        source_tokens: List of source tokens
        target_tokens: List of target tokens (optional, for labels)
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required!")
        return None
    
    if torch.is_tensor(attentions):
        attentions = attentions.detach().cpu().numpy()
    
    # Handle different input formats
    if len(attentions.shape) == 2:
        n_steps = attentions.shape[0]
        attn_list = [attentions[i] for i in range(n_steps)]
    else:
        attn_list = attentions
        n_steps = len(attn_list)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 4), sharey=True)
    
    if n_steps == 1:
        axes = [axes]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(source_tokens)))
    
    for i, (ax, attn) in enumerate(zip(axes, attn_list)):
        if torch.is_tensor(attn):
            attn = attn.detach().cpu().numpy()
        
        bars = ax.bar(range(len(attn)), attn, color=colors)
        
        # Highlight maximum
        max_idx = np.argmax(attn)
        bars[max_idx].set_color('darkblue')
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(2)
        
        # Labels
        step_label = target_tokens[i] if target_tokens else f"Step {i+1}"
        ax.set_title(f"→ {step_label}", fontsize=11)
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels(source_tokens, fontsize=9, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        if i == 0:
            ax.set_ylabel('Attention Weight', fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def plot_attention_comparison(attention_list, labels, source_tokens, target_idx=0,
                              title="Attention Comparison", save_path=None):
    """
    Compare attention distributions from different models or steps.
    
    Args:
        attention_list: List of attention vectors to compare
        labels: Labels for each attention vector
        source_tokens: Source tokens for x-axis
        target_idx: Which target position this is for
        title: Plot title
        save_path: Path to save
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(source_tokens))
    width = 0.8 / len(attention_list)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(attention_list)))
    
    for i, (attn, label, color) in enumerate(zip(attention_list, labels, colors)):
        if torch.is_tensor(attn):
            attn = attn.detach().cpu().numpy()
        offset = (i - len(attention_list) / 2 + 0.5) * width
        ax.bar(x + offset, attn, width, label=label, color=color, edgecolor='black')
    
    ax.set_xlabel('Source Position', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(source_tokens, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def attention_entropy(attention_weights, dim=-1):
    """
    Calculate entropy of attention distribution.
    
    Lower entropy = more focused attention
    Higher entropy = more spread out attention
    
    Args:
        attention_weights: Attention weights tensor/array
        dim: Dimension to compute entropy over
    
    Returns:
        Entropy value(s)
    """
    eps = 1e-9
    
    if torch.is_tensor(attention_weights):
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + eps), 
            dim=dim
        )
        return entropy
    else:
        entropy = -np.sum(
            attention_weights * np.log(attention_weights + eps),
            axis=dim
        )
        return entropy


def attention_coverage(attention_weights, threshold=0.1):
    """
    Calculate what fraction of source positions receive attention above threshold.
    
    Args:
        attention_weights: 2D attention matrix (target_len, source_len)
        threshold: Minimum attention weight to count
    
    Returns:
        Float between 0 and 1
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # How many source positions get significant attention
    covered = (attention_weights > threshold).any(axis=0).sum()
    total = attention_weights.shape[1]
    
    return covered / total


def check_reversal_pattern(attention_weights, tolerance=1):
    """
    Check if attention follows the expected reversal pattern.
    
    For sequence reversal, output i should attend to input (n-1-i).
    
    Args:
        attention_weights: 2D array (target_len, source_len)
        tolerance: How many positions off is acceptable
    
    Returns:
        Accuracy (0.0 to 1.0)
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    target_len, source_len = attention_weights.shape
    correct = 0
    
    for i in range(min(target_len, source_len)):
        peak = np.argmax(attention_weights[i])
        expected = source_len - 1 - i
        
        if abs(peak - expected) <= tolerance:
            correct += 1
    
    return correct / min(target_len, source_len)


def create_attention_report(model, dataset, device, num_samples=5, save_dir=None):
    """
    Generate a comprehensive attention analysis report.
    
    Args:
        model: Trained Seq2Seq model
        dataset: Dataset to sample from
        device: Torch device
        num_samples: Number of examples to analyze
        save_dir: Directory to save visualizations
    
    Returns:
        Dictionary with analysis results
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    results = {
        'examples': [],
        'avg_entropy': [],
        'reversal_accuracy': [],
        'coverage': []
    }
    
    print("\nAttention Analysis Report")
    print("=" * 60)
    
    for i in range(min(num_samples, len(dataset))):
        src, trg = dataset[i]
        src_t = src.unsqueeze(0).to(device)
        src_len = torch.tensor([len(src)]).to(device)
        
        with torch.no_grad():
            pred, attentions = model.translate(src_t, src_len)
        
        # Process outputs
        attn = attentions.squeeze(0).cpu().numpy()
        pred_list = pred.squeeze(0).cpu().tolist()
        
        # Remove EOS and after
        if 2 in pred_list:
            eos_idx = pred_list.index(2)
            pred_list = pred_list[:eos_idx]
            attn = attn[:eos_idx]
        
        target_list = trg[1:-1].tolist()
        correct = pred_list == target_list
        
        # Compute metrics
        avg_ent = float(np.mean(attention_entropy(attn, dim=-1)))
        rev_acc = check_reversal_pattern(attn)
        cov = attention_coverage(attn)
        
        results['examples'].append({
            'source': src.tolist(),
            'target': target_list,
            'prediction': pred_list,
            'attention': attn,
            'correct': correct,
            'entropy': avg_ent,
            'reversal_accuracy': rev_acc
        })
        results['avg_entropy'].append(avg_ent)
        results['reversal_accuracy'].append(rev_acc)
        results['coverage'].append(cov)
        
        # Print example
        status = "✓" if correct else "✗"
        print(f"\nExample {i+1}: {status}")
        print(f"  Source:     {src.tolist()}")
        print(f"  Prediction: {pred_list}")
        print(f"  Target:     {target_list}")
        print(f"  Entropy:    {avg_ent:.3f}")
        print(f"  Rev. Acc:   {rev_acc:.0%}")
        
        # Save visualization
        if save_dir and HAS_MATPLOTLIB:
            source_tokens = [str(t) for t in src.tolist()]
            target_tokens = [str(t) for t in pred_list]
            
            fig = plot_attention_heatmap(
                attn, source_tokens, target_tokens,
                title=f"Example {i+1}: {'Correct' if correct else 'Wrong'}",
                save_path=save_dir / f"attention_{i+1}.png"
            )
            plt.close(fig)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"  Average Entropy:      {np.mean(results['avg_entropy']):.3f}")
    print(f"  Reversal Accuracy:    {np.mean(results['reversal_accuracy']):.1%}")
    print(f"  Source Coverage:      {np.mean(results['coverage']):.1%}")
    print(f"  Prediction Accuracy:  {sum(e['correct'] for e in results['examples']) / len(results['examples']):.1%}")
    
    results['summary'] = {
        'avg_entropy': float(np.mean(results['avg_entropy'])),
        'avg_reversal_accuracy': float(np.mean(results['reversal_accuracy'])),
        'avg_coverage': float(np.mean(results['coverage'])),
        'prediction_accuracy': sum(e['correct'] for e in results['examples']) / len(results['examples'])
    }
    
    return results


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demo visualizations with synthetic data."""
    if not HAS_MATPLOTLIB:
        print("Install matplotlib for visualizations: pip install matplotlib")
        return
    
    print("Creating demo visualizations...")
    
    # Create synthetic perfect attention for reversal
    seq_len = 6
    source = ['5', '3', '8', '2', '1', '7']
    target = ['7', '1', '2', '8', '3', '5']
    
    # Perfect reversed diagonal
    attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        attention[i, seq_len - 1 - i] = 0.85
        # Add realistic noise
        for j in range(seq_len):
            if j != seq_len - 1 - i:
                attention[i, j] = 0.15 / (seq_len - 1)
    
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    # 1. Heatmap
    print("\n1. Creating attention heatmap...")
    fig1 = plot_attention_heatmap(
        attention, source, target,
        title="Sequence Reversal: Perfect Attention Pattern"
    )
    plt.show()
    
    # 2. Evolution
    print("\n2. Creating attention evolution...")
    fig2 = plot_attention_evolution(
        attention, source, target,
        title="How Attention Shifts During Decoding"
    )
    plt.show()
    
    # 3. Analysis
    print("\n3. Attention Analysis:")
    print(f"   Average Entropy: {np.mean(attention_entropy(attention, dim=-1)):.3f}")
    print(f"   Reversal Pattern: {check_reversal_pattern(attention):.0%}")
    print(f"   Source Coverage: {attention_coverage(attention):.0%}")
    
    print("\n✓ Demo complete!")


if __name__ == '__main__':
    demo()
