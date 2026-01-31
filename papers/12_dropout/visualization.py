"""
Dropout Visualizations

Create compelling visualizations to understand dropout mechanics.

Visualizations:
1. Training curves: With vs Without Dropout
2. Dropout mask patterns
3. Feature redundancy comparison
4. Network ensemble interpretation
5. MC Dropout uncertainty
6. Dropout rate sweep results

Author: 30u30 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional
import os


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11


def plot_training_curves_comparison(
    history_with_dropout: dict,
    history_without_dropout: dict,
    save_path: Optional[str] = None
):
    """
    Compare training curves with and without dropout.
    
    Shows the overfitting gap reduction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history_without_dropout['train_acc']) + 1)
    
    # Without Dropout
    ax = axes[0]
    ax.plot(epochs, history_without_dropout['train_acc'], 'b-', 
            linewidth=2, label='Train')
    ax.plot(epochs, history_without_dropout['test_acc'], 'r-', 
            linewidth=2, label='Test')
    
    # Fill the gap
    ax.fill_between(epochs, 
                    history_without_dropout['train_acc'],
                    history_without_dropout['test_acc'],
                    alpha=0.3, color='red', label='Overfitting Gap')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Without Dropout: Large Gap = Overfitting', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.05)
    
    # Add gap annotation
    final_gap = history_without_dropout['train_acc'][-1] - history_without_dropout['test_acc'][-1]
    ax.annotate(f'Gap: {final_gap:.1%}', 
                xy=(len(epochs), np.mean([history_without_dropout['train_acc'][-1], 
                                          history_without_dropout['test_acc'][-1]])),
                fontsize=12, fontweight='bold', color='red')
    
    # With Dropout
    ax = axes[1]
    ax.plot(epochs, history_with_dropout['train_acc'], 'b-', 
            linewidth=2, label='Train')
    ax.plot(epochs, history_with_dropout['test_acc'], 'g-', 
            linewidth=2, label='Test')
    
    ax.fill_between(epochs,
                    history_with_dropout['train_acc'],
                    history_with_dropout['test_acc'],
                    alpha=0.3, color='green', label='Small Gap')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('With Dropout (p=0.5): Small Gap = Generalization', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.05)
    
    # Add gap annotation
    final_gap = history_with_dropout['train_acc'][-1] - history_with_dropout['test_acc'][-1]
    ax.annotate(f'Gap: {final_gap:.1%}',
                xy=(len(epochs), np.mean([history_with_dropout['train_acc'][-1],
                                          history_with_dropout['test_acc'][-1]])),
                fontsize=12, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_dropout_masks(p: float = 0.5, n_masks: int = 10, save_path: Optional[str] = None):
    """
    Visualize different dropout masks.
    
    Shows how each forward pass uses a different random subset.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, ax in enumerate(axes.flatten()):
        # Generate mask
        mask = (np.random.rand(8, 8) < p).astype(np.float32)
        
        # Plot
        cmap = LinearSegmentedColormap.from_list('dropout', ['#ff4444', '#44ff44'])
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        
        kept = mask.sum()
        total = mask.size
        ax.set_title(f'Mask {i+1}\n{int(kept)}/{total} kept ({100*kept/total:.0f}%)', fontsize=10)
        ax.axis('off')
        
        # Add grid lines
        for x in range(8):
            ax.axhline(y=x-0.5, color='white', linewidth=0.5)
            ax.axvline(x=x-0.5, color='white', linewidth=0.5)
    
    plt.suptitle(f'Dropout Masks (keep probability p={p})\nEach forward pass uses a different random mask!', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#44ff44', label='Kept (active)'),
        mpatches.Patch(facecolor='#ff4444', label='Dropped (zeroed)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_feature_redundancy(save_path: Optional[str] = None):
    """
    Show how dropout encourages redundant features.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Without dropout: Sparse, fragile features
    np.random.seed(42)
    no_dropout = np.zeros((6, 6))
    no_dropout[0, 0] = 0.95
    no_dropout[1, 1] = 0.1
    no_dropout += np.random.rand(6, 6) * 0.05
    
    ax = axes[0]
    im = ax.imshow(no_dropout, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('Without Dropout: Sparse Features\n(If one neuron fails, recognition fails!)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Input Features')
    ax.set_ylabel('Hidden Neurons')
    plt.colorbar(im, ax=ax, label='Activation Strength')
    
    # With dropout: Distributed, robust features
    with_dropout = np.random.rand(6, 6) * 0.5 + 0.3
    with_dropout = with_dropout / with_dropout.sum(axis=0, keepdims=True)
    
    ax = axes[1]
    im = ax.imshow(with_dropout, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('With Dropout: Distributed Features\n(Redundancy = Robustness)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Input Features')
    ax.set_ylabel('Hidden Neurons')
    plt.colorbar(im, ax=ax, label='Activation Strength')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_ensemble_interpretation(save_path: Optional[str] = None):
    """
    Visualize dropout as an ensemble of networks.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    
    # Title
    ax.text(5, 7.5, 'Dropout = Implicit Ensemble of Networks', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Draw multiple networks
    networks_y = [5.5, 3.5, 1.5]
    networks_masks = [
        [1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0]
    ]
    
    for y, mask in zip(networks_y, networks_masks):
        # Draw network box
        rect = plt.Rectangle((0.5, y-0.4), 9, 0.8, 
                             fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Draw neurons
        for i, keep in enumerate(mask):
            x = 1.5 + i * 1.4
            color = '#44ff44' if keep else '#ff4444'
            alpha = 1.0 if keep else 0.3
            circle = plt.Circle((x, y), 0.25, color=color, alpha=alpha)
            ax.add_patch(circle)
        
        # Label
        ax.text(10, y, f'Network {int((5.5-y)//2)+1}', fontsize=11, va='center')
    
    # Arrow to final prediction
    ax.annotate('', xy=(5, 0.5), xytext=(5, 1.3),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.text(5, 0.2, 'Average prediction (ensemble)', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Explanation
    ax.text(5, -0.5, 
            'Each forward pass samples a different "sub-network"\n'
            'Test time uses the average of all possible networks',
            fontsize=11, ha='center', style='italic')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#44ff44', label='Active neuron'),
        mpatches.Patch(facecolor='#ff4444', alpha=0.3, label='Dropped neuron')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_dropout_rate_sweep(
    results: dict,
    save_path: Optional[str] = None
):
    """
    Plot results from sweeping dropout rates.
    
    Args:
        results: Dict with dropout rates as keys, containing train_acc, test_acc, gap
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rates = sorted(results.keys())
    train_accs = [results[r]['train_acc'] for r in rates]
    test_accs = [results[r]['test_acc'] for r in rates]
    gaps = [results[r]['gap'] for r in rates]
    
    # Left: Accuracies
    ax = axes[0]
    ax.plot(rates, train_accs, 'bo-', linewidth=2, markersize=8, label='Train Accuracy')
    ax.plot(rates, test_accs, 'go-', linewidth=2, markersize=8, label='Test Accuracy')
    
    ax.set_xlabel('Dropout Keep Probability (p)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Dropout Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Highlight best test accuracy
    best_idx = np.argmax(test_accs)
    ax.axvline(x=rates[best_idx], color='green', linestyle='--', alpha=0.5)
    ax.annotate(f'Best: p={rates[best_idx]:.1f}',
                xy=(rates[best_idx], test_accs[best_idx]),
                xytext=(rates[best_idx]+0.15, test_accs[best_idx]),
                fontsize=10, color='green')
    
    # Right: Gap
    ax = axes[1]
    colors = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green' for g in gaps]
    bars = ax.bar(rates, gaps, width=0.08, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Dropout Keep Probability (p)', fontsize=12)
    ax.set_ylabel('Train-Test Gap', fontsize=12)
    ax.set_title('Overfitting Gap vs Dropout Rate', fontsize=14, fontweight='bold')
    ax.axhline(y=0.05, color='green', linestyle='--', label='Good (<5%)', alpha=0.7)
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Acceptable (<10%)', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Add gap indicator
    for i, (rate, gap) in enumerate(zip(rates, gaps)):
        symbol = '✓' if gap < 0.05 else ('⚠' if gap < 0.1 else '✗')
        ax.text(rate, gap + 0.02, symbol, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_mc_dropout_uncertainty(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    n_samples: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize MC Dropout uncertainty estimation.
    
    Args:
        predictions: (n_forward_passes, n_samples, n_classes)
        true_labels: (n_samples,)
    """
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    
    for idx, ax in enumerate(axes.flatten()):
        if idx >= n_samples:
            ax.axis('off')
            continue
        
        sample_preds = predictions[:, idx, :]  # (n_passes, n_classes)
        
        mean_pred = sample_preds.mean(axis=0)
        std_pred = sample_preds.std(axis=0)
        
        predicted_class = mean_pred.argmax()
        true_class = true_labels[idx]
        uncertainty = std_pred.mean()
        
        # Bar plot with error bars
        x = np.arange(10)
        bars = ax.bar(x, mean_pred, yerr=std_pred, 
                     color=['#44ff44' if i == true_class else '#4444ff' for i in range(10)],
                     capsize=3, alpha=0.8)
        
        # Highlight predicted class
        ax.bar(predicted_class, mean_pred[predicted_class], 
               color='red' if predicted_class != true_class else '#44ff44',
               edgecolor='red', linewidth=2)
        
        correct = '✓' if predicted_class == true_class else '✗'
        ax.set_title(f'Sample {idx+1} {correct}\n'
                     f'Pred: {predicted_class}, True: {true_class}\n'
                     f'Uncertainty: {uncertainty:.3f}',
                     fontsize=10)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
    
    plt.suptitle('MC Dropout: Multiple Forward Passes with Dropout ON\n'
                 'Error bars show uncertainty from multiple predictions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_inverted_vs_naive_dropout(save_path: Optional[str] = None):
    """
    Compare inverted dropout vs naive dropout scaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Naive dropout visualization
    ax = axes[0]
    ax.set_title('Naive Dropout (Original)', fontsize=12, fontweight='bold')
    
    # Training
    ax.text(0.1, 0.9, 'Training:', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.8, 'output = input × mask', fontsize=10, transform=ax.transAxes,
            family='monospace')
    ax.text(0.1, 0.7, '(No scaling)', fontsize=9, transform=ax.transAxes, style='italic')
    
    # Testing
    ax.text(0.1, 0.5, 'Testing:', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.4, 'output = input × p', fontsize=10, transform=ax.transAxes,
            family='monospace', color='red')
    ax.text(0.1, 0.3, '(MUST remember to scale!)', fontsize=9, transform=ax.transAxes, 
            style='italic', color='red')
    
    ax.text(0.1, 0.1, '⚠ Easy to forget test-time scaling', fontsize=10, transform=ax.transAxes,
            color='orange')
    ax.axis('off')
    
    # Inverted dropout visualization  
    ax = axes[1]
    ax.set_title('Inverted Dropout (Modern)', fontsize=12, fontweight='bold')
    
    # Training
    ax.text(0.1, 0.9, 'Training:', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.8, 'output = (input × mask) / p', fontsize=10, transform=ax.transAxes,
            family='monospace')
    ax.text(0.1, 0.7, '(Scale UP during training)', fontsize=9, transform=ax.transAxes, 
            style='italic', color='blue')
    
    # Testing
    ax.text(0.1, 0.5, 'Testing:', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.4, 'output = input', fontsize=10, transform=ax.transAxes,
            family='monospace', color='green')
    ax.text(0.1, 0.3, '(No change needed!)', fontsize=9, transform=ax.transAxes, 
            style='italic', color='green')
    
    ax.text(0.1, 0.1, '✓ Simpler and safer', fontsize=10, transform=ax.transAxes,
            color='green')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_all_visualizations(output_dir: str = 'visualizations'):
    """Generate all visualizations and save to directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Dropout masks
    plot_dropout_masks(p=0.5, save_path=os.path.join(output_dir, 'dropout_masks.png'))
    
    # 2. Feature redundancy
    plot_feature_redundancy(save_path=os.path.join(output_dir, 'feature_redundancy.png'))
    
    # 3. Ensemble interpretation
    plot_ensemble_interpretation(save_path=os.path.join(output_dir, 'ensemble_interpretation.png'))
    
    # 4. Inverted vs naive
    plot_inverted_vs_naive_dropout(save_path=os.path.join(output_dir, 'inverted_vs_naive.png'))
    
    # 5. Simulated training curves
    np.random.seed(42)
    epochs = 50
    
    # Without dropout: overfits
    no_dropout_train = 0.5 + 0.5 * (1 - np.exp(-np.arange(epochs) / 10))
    no_dropout_test = 0.5 + 0.25 * np.log1p(np.arange(epochs) / 5)
    no_dropout_test = np.minimum(no_dropout_test, 0.75)
    
    # With dropout: generalizes
    dropout_train = 0.5 + 0.4 * (1 - np.exp(-np.arange(epochs) / 15))
    dropout_test = 0.5 + 0.35 * (1 - np.exp(-np.arange(epochs) / 20))
    
    history_no_dropout = {'train_acc': no_dropout_train, 'test_acc': no_dropout_test}
    history_dropout = {'train_acc': dropout_train, 'test_acc': dropout_test}
    
    plot_training_curves_comparison(
        history_dropout, history_no_dropout,
        save_path=os.path.join(output_dir, 'training_curves.png')
    )
    
    # 6. Rate sweep
    results = {
        0.0: {'train_acc': 0.99, 'test_acc': 0.65, 'gap': 0.34},
        0.3: {'train_acc': 0.95, 'test_acc': 0.78, 'gap': 0.17},
        0.5: {'train_acc': 0.90, 'test_acc': 0.85, 'gap': 0.05},
        0.7: {'train_acc': 0.82, 'test_acc': 0.80, 'gap': 0.02},
        0.9: {'train_acc': 0.60, 'test_acc': 0.58, 'gap': 0.02},
    }
    plot_dropout_rate_sweep(results, save_path=os.path.join(output_dir, 'rate_sweep.png'))
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_all_visualizations()
