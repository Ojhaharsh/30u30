"""
Visualization Tools for Regularization
=======================================

Visualize the effects of regularization:
- Training vs validation loss curves
- Dropout effect on activations
- Weight distributions
- Gradient norms during training
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_learning_curves(train_losses: List[float], val_losses: List[float],
                        title: str = "Learning Curves", save_path: str = None):
    """
    Plot training and validation loss curves.
    
    Shows the classic pattern:
    - Training loss: steady decrease
    - Validation loss: decreases then plateaus or increases (overfitting)
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(len(train_losses))
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    # Find divergence point
    if len(val_losses) > 1:
        best_epoch = np.argmin(val_losses)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch: {best_epoch}')
        ax.scatter([best_epoch], [val_losses[best_epoch]], color='green', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_regularization_comparison(results: Dict[str, Dict[str, List[float]]],
                                   save_path: str = None):
    """
    Compare learning curves with and without regularization.
    
    Shows how regularization prevents overfitting.
    
    Args:
        results: Dictionary with structure:
                 {'without_reg': {'train': [...], 'val': [...]},
                  'with_reg': {'train': [...], 'val': [...]}}
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Without regularization
    train_no_reg = results['without_reg']['train']
    val_no_reg = results['without_reg']['val']
    epochs_no_reg = range(len(train_no_reg))
    
    axes[0].plot(epochs_no_reg, train_no_reg, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    axes[0].plot(epochs_no_reg, val_no_reg, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    axes[0].fill_between(epochs_no_reg, train_no_reg, val_no_reg, alpha=0.2, color='red', label='Overfitting Gap')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('WITHOUT Regularization\n(Overfitting)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # With regularization
    train_reg = results['with_reg']['train']
    val_reg = results['with_reg']['val']
    epochs_reg = range(len(train_reg))
    
    axes[1].plot(epochs_reg, train_reg, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    axes[1].plot(epochs_reg, val_reg, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    axes[1].fill_between(epochs_reg, train_reg, val_reg, alpha=0.2, color='green', label='Small Gap')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('WITH Regularization\n(Controlled)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Match y-axis scales
    all_losses = train_no_reg + val_no_reg + train_reg + val_reg
    y_max = max(all_losses) * 1.1
    axes[0].set_ylim([0, y_max])
    axes[1].set_ylim([0, y_max])
    
    fig.suptitle('Regularization Effect: Controlling Overfitting', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dropout_effect(activations_no_dropout: np.ndarray,
                        activations_with_dropout: np.ndarray,
                        save_path: str = None):
    """
    Visualize dropout's effect on activations.
    
    Shows how dropout randomly deactivates neurons.
    
    Args:
        activations_no_dropout: Activations without dropout
        activations_with_dropout: Activations with dropout
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Without dropout
    im1 = axes[0].imshow(activations_no_dropout, cmap='RdYlGn', aspect='auto')
    axes[0].set_title('Without Dropout\n(All neurons active)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Hidden units')
    axes[0].set_ylabel('Time steps')
    plt.colorbar(im1, ax=axes[0])
    
    # With dropout
    im2 = axes[1].imshow(activations_with_dropout, cmap='RdYlGn', aspect='auto')
    axes[1].set_title('With Dropout (keep_prob=0.8)\n(20% randomly deactivated)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hidden units')
    axes[1].set_ylabel('Time steps')
    plt.colorbar(im2, ax=axes[1])
    
    fig.suptitle('Dropout: Randomly Disabling Neurons', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_weight_distributions(weights_initial: np.ndarray,
                              weights_trained: np.ndarray,
                              weights_regularized: np.ndarray,
                              save_path: str = None):
    """
    Plot weight distributions showing effect of regularization.
    
    Weight decay pulls weights toward zero.
    
    Args:
        weights_initial: Initial weight distribution
        weights_trained: Trained without regularization
        weights_regularized: Trained with regularization (weight decay)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Initial weights
    axes[0].hist(weights_initial.flatten(), bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Initial Weights\n(Small random values)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Weight value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Trained without regularization
    axes[1].hist(weights_trained.flatten(), bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_title('After Training (No Regularization)\n(Large spread)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Weight value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Trained with regularization
    axes[2].hist(weights_regularized.flatten(), bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[2].set_title('After Training (With Weight Decay)\n(Concentrated near 0)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Weight value')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    fig.suptitle('Weight Decay: Pulling Weights Toward Zero', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gradient_norms(gradient_norms: List[float],
                       clipped_gradient_norms: List[float] = None,
                       save_path: str = None):
    """
    Plot gradient norms during training.
    
    Shows gradient explosion and the effect of clipping.
    
    Args:
        gradient_norms: List of gradient norms per step
        clipped_gradient_norms: Gradient norms after clipping (optional)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = range(len(gradient_norms))
    
    ax.plot(steps, gradient_norms, 'b-', linewidth=1.5, label='Original gradients', alpha=0.8)
    
    if clipped_gradient_norms is not None:
        ax.plot(steps, clipped_gradient_norms, 'r-', linewidth=1.5, label='Clipped gradients', alpha=0.8)
    
    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel('Gradient norm', fontsize=12)
    ax.set_title('Gradient Norms During Training\n(Gradient clipping prevents explosion)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see small gradients
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_overfitting_vs_regularization_strength(regularization_strengths: List[float],
                                               train_losses: List[float],
                                               val_losses: List[float],
                                               save_path: str = None):
    """
    Show the relationship between regularization strength and overfitting.
    
    Too little regularization: overfitting
    Too much regularization: underfitting
    
    Args:
        regularization_strengths: List of lambda values (weight decay strength)
        train_losses: Training losses
        val_losses: Validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(regularization_strengths, train_losses, 'b-o', linewidth=2, label='Training Loss', markersize=8)
    ax.plot(regularization_strengths, val_losses, 'r-s', linewidth=2, label='Validation Loss', markersize=8)
    
    # Shade overfitting region (left) and underfitting region (right)
    mid = len(regularization_strengths) // 2
    ax.axvspan(regularization_strengths[0], regularization_strengths[mid], alpha=0.1, color='red', label='Overfitting region')
    ax.axvspan(regularization_strengths[mid], regularization_strengths[-1], alpha=0.1, color='blue', label='Underfitting region')
    
    ax.set_xlabel('Regularization Strength (λ)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Regularization Strength vs Overfitting\n(Sweet spot in the middle)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Functions available:")
    print("  - plot_learning_curves()")
    print("  - plot_regularization_comparison()")
    print("  - plot_dropout_effect()")
    print("  - plot_weight_distributions()")
    print("  - plot_gradient_norms()")
    print("  - plot_overfitting_vs_regularization_strength()")
    print("  - plot_layer_norm_effect()")
    print("  - plot_early_stopping_analysis()")
    print("  - create_regularization_summary_figure()")


def plot_layer_norm_effect(activations_before: np.ndarray,
                           activations_after: np.ndarray,
                           save_path: str = None):
    """
    Visualize layer normalization's effect on activations.
    
    Shows how layer norm standardizes activations to zero mean and unit variance.
    
    Args:
        activations_before: Activations before normalization
        activations_after: Activations after normalization
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Before normalization - heatmap
    im1 = axes[0, 0].imshow(activations_before, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Before Layer Norm (Heatmap)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Hidden units')
    axes[0, 0].set_ylabel('Time steps')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # After normalization - heatmap
    im2 = axes[0, 1].imshow(activations_after, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('After Layer Norm (Heatmap)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Hidden units')
    axes[0, 1].set_ylabel('Time steps')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Before normalization - histogram
    axes[1, 0].hist(activations_before.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
    mean_before = np.mean(activations_before)
    std_before = np.std(activations_before)
    axes[1, 0].axvline(x=mean_before, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_before:.2f}')
    axes[1, 0].set_title(f'Before Layer Norm (Distribution)\nMean: {mean_before:.2f}, Std: {std_before:.2f}', 
                         fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Activation value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # After normalization - histogram
    axes[1, 1].hist(activations_after.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    mean_after = np.mean(activations_after)
    std_after = np.std(activations_after)
    axes[1, 1].axvline(x=mean_after, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_after:.2f}')
    axes[1, 1].set_title(f'After Layer Norm (Distribution)\nMean: {mean_after:.2f}, Std: {std_after:.2f}', 
                         fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Activation value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Layer Normalization: Standardizing Activations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_early_stopping_analysis(train_losses: List[float],
                                 val_losses: List[float],
                                 stopped_epoch: int = None,
                                 best_epoch: int = None,
                                 save_path: str = None):
    """
    Visualize early stopping decision making.
    
    Shows when validation loss stopped improving and when training stopped.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        stopped_epoch: Epoch when training stopped
        best_epoch: Epoch with best validation loss
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = range(len(train_losses))
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=6)
    ax.plot(epochs, val_losses, 'r-', linewidth=2.5, label='Validation Loss', marker='s', markersize=6)
    
    # Mark best epoch
    if best_epoch is not None and best_epoch < len(val_losses):
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'Best epoch: {best_epoch}')
        ax.scatter([best_epoch], [val_losses[best_epoch]], color='green', s=200, zorder=5, marker='*', 
                   edgecolor='black', linewidth=2)
        ax.annotate(f'Best val loss: {val_losses[best_epoch]:.4f}', 
                    xy=(best_epoch, val_losses[best_epoch]),
                    xytext=(best_epoch + 1, val_losses[best_epoch] - 0.1),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Mark stopped epoch
    if stopped_epoch is not None and stopped_epoch < len(val_losses):
        ax.axvline(x=stopped_epoch, color='red', linestyle=':', linewidth=2, alpha=0.8, label=f'Stopped at: {stopped_epoch}')
        
        # Shade the "overfitting zone"
        if best_epoch is not None:
            ax.axvspan(best_epoch, stopped_epoch, alpha=0.1, color='red', label='Overfitting zone')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Early Stopping Analysis\n(Stop before overfitting gets worse)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def create_regularization_summary_figure(results: Dict,
                                         save_path: str = None):
    """
    Create a comprehensive summary figure of all regularization techniques.
    
    Args:
        results: Dictionary with keys for each technique and their effects
                 Expected structure:
                 {
                     'dropout': {'activations_on': ndarray, 'activations_off': ndarray},
                     'layer_norm': {'before': ndarray, 'after': ndarray},
                     'weight_decay': {'weights_no_reg': ndarray, 'weights_reg': ndarray},
                     'learning_curves': {'train': list, 'val': list}
                 }
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Learning curves (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'learning_curves' in results:
        train = results['learning_curves']['train']
        val = results['learning_curves']['val']
        ax1.plot(train, 'b-', linewidth=2, label='Training')
        ax1.plot(val, 'r-', linewidth=2, label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Dropout effect (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'dropout' in results:
        dropout_on = np.sum(results['dropout']['activations_on'] == 0, axis=1)
        dropout_off = np.sum(results['dropout']['activations_off'] == 0, axis=1)
        ax2.bar(['No Dropout', 'With Dropout'], [np.mean(dropout_off), np.mean(dropout_on)], 
                color=['red', 'green'], edgecolor='black')
        ax2.set_ylabel('Zero activations')
        ax2.set_title('Dropout Effect', fontweight='bold')
    
    # 3. Weight distributions (middle row)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'weight_decay' in results:
        ax3.hist(results['weight_decay']['weights_no_reg'].flatten(), bins=30, 
                 color='red', alpha=0.7, label='No regularization')
        ax3.hist(results['weight_decay']['weights_reg'].flatten(), bins=30, 
                 color='green', alpha=0.7, label='With weight decay')
        ax3.set_xlabel('Weight value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Weight Distributions', fontweight='bold')
        ax3.legend()
    
    # 4. Layer norm effect (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'layer_norm' in results:
        before_std = np.std(results['layer_norm']['before'])
        after_std = np.std(results['layer_norm']['after'])
        ax4.bar(['Before LayerNorm', 'After LayerNorm'], [before_std, after_std],
                color=['red', 'green'], edgecolor='black')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Layer Norm Standardization', fontweight='bold')
    
    # 5. Summary metrics (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    techniques = ['Dropout', 'LayerNorm', 'Weight Decay', 'Early Stop']
    effects = [0.8, 0.9, 0.7, 0.85]  # Placeholder effectiveness scores
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax5.barh(techniques, effects, color=colors, edgecolor='black')
    ax5.set_xlabel('Relative Effectiveness')
    ax5.set_title('Technique Comparison', fontweight='bold')
    ax5.set_xlim([0, 1])
    
    # 6. Summary text (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    summary_text = """
    REGULARIZATION SUMMARY
    ═══════════════════════
    
    ✓ Dropout:       Randomly disables neurons to prevent co-adaptation
    ✓ Layer Norm:    Standardizes activations for stable training  
    ✓ Weight Decay:  Penalizes large weights to encourage simplicity
    ✓ Early Stopping: Stops training before memorization occurs
    
    Use ALL FOUR techniques together for best results!
    """
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    fig.suptitle('Regularization Techniques: Complete Overview', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def demo_visualizations():
    """
    Generate sample visualizations with synthetic data.
    
    Useful for testing and demonstration.
    """
    print("Generating sample visualizations...")
    
    # Generate synthetic data
    np.random.seed(42)
    
    # 1. Learning curves
    epochs = 50
    train_losses = 2.5 * np.exp(-np.arange(epochs) / 15) + 0.3 + np.random.randn(epochs) * 0.05
    val_losses = 2.5 * np.exp(-np.arange(epochs) / 20) + 0.5 + np.random.randn(epochs) * 0.05
    val_losses[30:] += np.linspace(0, 0.5, 20)  # Add overfitting
    
    fig1 = plot_learning_curves(train_losses.tolist(), val_losses.tolist(),
                                title="Example Learning Curves")
    print("  ✓ Learning curves")
    
    # 2. Dropout effect
    activations_no_dropout = np.random.randn(20, 10)
    dropout_mask = np.random.binomial(1, 0.8, (20, 10))
    activations_with_dropout = activations_no_dropout * dropout_mask
    
    fig2 = plot_dropout_effect(activations_no_dropout, activations_with_dropout)
    print("  ✓ Dropout effect")
    
    # 3. Weight distributions
    weights_initial = np.random.randn(100, 100) * 0.1
    weights_trained = np.random.randn(100, 100) * 0.5
    weights_regularized = np.random.randn(100, 100) * 0.2
    
    fig3 = plot_weight_distributions(weights_initial, weights_trained, weights_regularized)
    print("  ✓ Weight distributions")
    
    # 4. Early stopping
    fig4 = plot_early_stopping_analysis(train_losses.tolist(), val_losses.tolist(),
                                        stopped_epoch=35, best_epoch=25)
    print("  ✓ Early stopping analysis")
    
    print("\nAll visualizations generated! Close windows to exit.")
    plt.show()
