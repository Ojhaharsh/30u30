"""
Day 15: Visualization Utilities for Bahdanau Attention

Technical utilities for analyzing and visualizing attention distributions
in sequence-to-sequence models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_attention_heatmap(
    attention_weights: Union[torch.Tensor, np.ndarray],
    source_tokens: List[str],
    target_tokens: List[str],
    title: str = "Attention Weights Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = 'Blues',
    show_values: bool = True
):
    """
    Visualize attention weights as a heatmap matrix.
    
    Args:
        attention_weights: [target_len, source_len] weight matrix.
        source_tokens: tokens for the x-axis.
        target_tokens: tokens for the y-axis.
        title: chart title.
        save_path: optional path to save the generated figure.
        figsize: figure dimensions.
        cmap: matplotlib colormap.
        show_values: if True, annotate cells with numerical weights.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for attention visualization.")
        return None
    
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=12)
    
    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens, fontsize=10)
    
    ax.set_xlabel('Source Sequence Index', fontsize=12)
    ax.set_ylabel('Target Sequence Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if show_values:
        for i in range(len(target_tokens)):
            for j in range(len(source_tokens)):
                val = attention_weights[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_evolution(
    attentions: Union[torch.Tensor, np.ndarray],
    source_tokens: List[str],
    target_tokens: Optional[List[str]] = None,
    title: str = "Attention Evolution per Decoding Step",
    save_path: Optional[str] = None
):
    """
    Visualize the shift in attention distribution across decoding steps.
    
    Args:
        attentions: [target_len, source_len] matrix.
        source_tokens: labels for the x-axis.
        target_tokens: optional labels for each decoding step.
        title: chart title.
        save_path: optional save destination.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if torch.is_tensor(attentions):
        attentions = attentions.detach().cpu().numpy()
    
    n_steps = attentions.shape[0]
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 4), sharey=True)
    
    if n_steps == 1:
        axes = [axes]
    
    for i, (ax, attn) in enumerate(zip(axes, attentions)):
        ax.bar(range(len(attn)), attn, color='steelblue', edgecolor='black')
        step_label = target_tokens[i] if target_tokens else f"Step {i+1}"
        ax.set_title(step_label, fontsize=11)
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels(source_tokens, fontsize=9, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def calculate_attention_entropy(attention_weights: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute the Shannon entropy of the attention distribution.
    Lower values indicate higher focus.
    """
    eps = 1e-9
    if torch.is_tensor(attention_weights):
        return -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
    return -np.sum(attention_weights * np.log(attention_weights + eps), axis=-1)


def create_attention_analysis(model, dataset, device, num_samples=5, save_dir=None):
    """Generates attention metrics and saved plots for a given model and dataset."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_metrics = []
    
    for i in range(min(num_samples, len(dataset))):
        src, trg = dataset[i]
        src_t = src.unsqueeze(0).to(device)
        src_len = torch.tensor([len(src)]).to(device)
        
        with torch.no_grad():
            pred, attractions = model.translate(src_t, src_len)
        
        attn = attractions.squeeze(0).cpu().numpy()
        pred_list = pred.squeeze(0).cpu().tolist()
        
        if 2 in pred_list:
            eos_idx = pred_list.index(2)
            attn = attn[:eos_idx]
        
        entropy = calculate_attention_entropy(attn)
        all_metrics.append(np.mean(entropy))
        
        if save_dir and HAS_MATPLOTLIB:
            source_tokens = [str(t) for t in src.tolist()]
            target_tokens = [str(t) for t in (pred_list[:eos_idx] if 2 in pred_list else pred_list)]
            plot_attention_heatmap(
                attn, source_tokens, target_tokens,
                save_path=save_dir / f"attention_sample_{i+1}.png"
            )
            
    return {"avg_entropy": np.mean(all_metrics)}


if __name__ == '__main__':
    print("Attention visualization module.")
    # Add minimal demo if needed, otherwise leave as utility.
