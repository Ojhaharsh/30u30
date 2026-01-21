"""
Visualization Tools for Character-Level RNN
===========================================

Functions to visualize:
1. Training loss curves
2. Hidden state evolution
3. Character probability heatmaps
4. Gradient flow through time
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(losses, title="Training Loss", save_path=None):
    """
    Plot training loss over time.
    
    Args:
        losses: List of loss values
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add smoothed curve
    if len(losses) > 100:
        window = 100
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), smoothed, 
                color='red', linewidth=2, label='Smoothed', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_hidden_states(hidden_states, char_sequence, save_path=None):
    """
    Visualize how hidden states evolve over a sequence.
    
    Args:
        hidden_states: Array of shape (seq_len, hidden_size)
        char_sequence: String of characters corresponding to each time step
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Transpose so rows are hidden units, columns are time steps
    hidden_states_T = hidden_states.T
    
    sns.heatmap(hidden_states_T, cmap='RdBu_r', center=0, 
                xticklabels=list(char_sequence),
                yticklabels=False,
                cbar_kws={'label': 'Activation'})
    
    plt.xlabel('Character', fontsize=12)
    plt.ylabel('Hidden Unit', fontsize=12)
    plt.title('Hidden State Evolution Over Sequence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_probability_distribution(probs, chars, top_k=10, save_path=None):
    """
    Plot probability distribution over characters at a single time step.
    
    Args:
        probs: Array of probabilities (vocab_size,)
        chars: List of characters
        top_k: Number of top characters to show
        save_path: Optional path to save figure
    """
    # Get top-k characters
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_chars = [chars[i] for i in top_indices]
    
    # Replace special characters for display
    display_chars = []
    for ch in top_chars:
        if ch == '\n':
            display_chars.append('\\n')
        elif ch == ' ':
            display_chars.append('SPACE')
        else:
            display_chars.append(ch)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(top_k), top_probs, color='skyblue', edgecolor='navy')
    
    # Highlight the top prediction
    bars[0].set_color('coral')
    
    plt.xlabel('Character', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Character Probability Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(top_k), display_chars, fontsize=10)
    plt.ylim(0, max(top_probs) * 1.1)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_sequence_predictions(text, predictions, chars, save_path=None):
    """
    Show model predictions for each character in a sequence.
    
    Args:
        text: Input text string
        predictions: Array of shape (seq_len, vocab_size) with probabilities
        chars: List of all characters in vocabulary
        save_path: Optional path to save figure
    """
    seq_len = len(text)
    
    # Get top-3 predictions at each step
    fig, axes = plt.subplots(seq_len, 1, figsize=(12, 2*seq_len))
    
    if seq_len == 1:
        axes = [axes]
    
    for t, (char, probs, ax) in enumerate(zip(text, predictions, axes)):
        # Get top-3
        top_indices = np.argsort(probs)[-3:][::-1]
        top_probs = probs[top_indices]
        top_chars = [chars[i] for i in top_indices]
        
        # Display
        display_chars = [ch if ch not in ['\n', ' '] else ('\\n' if ch == '\n' else 'SPACE') for ch in top_chars]
        
        bars = ax.barh(range(3), top_probs, color=['green', 'orange', 'red'])
        ax.set_yticks(range(3))
        ax.set_yticklabels(display_chars)
        ax.set_xlim(0, 1)
        ax.set_title(f"After '{char}' → Predict next", fontsize=10)
        ax.set_xlabel('Probability')
        
        # Add value labels
        for bar, prob in zip(bars, top_probs):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{prob:.3f}',
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_gradient_flow(grad_norms, save_path=None):
    """
    Visualize gradient magnitudes through time (for diagnosing vanishing/exploding gradients).
    
    Args:
        grad_norms: Array of gradient norms at each time step
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    time_steps = range(len(grad_norms))
    plt.plot(time_steps, grad_norms, marker='o', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='green', linestyle='--', label='Gradient norm = 1', alpha=0.7)
    plt.axhline(y=0.1, color='orange', linestyle='--', label='Vanishing threshold', alpha=0.7)
    plt.axhline(y=10.0, color='red', linestyle='--', label='Exploding threshold', alpha=0.7)
    
    plt.xlabel('Time Step (backward)', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title('Gradient Flow Through Time', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_temperature_comparison(samples_dict, save_path=None):
    """
    Compare samples generated at different temperatures.
    
    Args:
        samples_dict: Dict mapping temperature -> generated text
        save_path: Optional path to save figure
    """
    temps = sorted(samples_dict.keys())
    n_temps = len(temps)
    
    fig, ax = plt.subplots(figsize=(12, max(8, n_temps * 1.5)))
    ax.axis('off')
    
    title = "Effect of Temperature on Sampling"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Create text display
    y_pos = 0.95
    for temp in temps:
        sample = samples_dict[temp]
        
        # Truncate if too long
        if len(sample) > 200:
            sample = sample[:200] + "..."
        
        # Add temperature label
        ax.text(0.05, y_pos, f"Temperature = {temp:.1f}:",
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.03
        
        # Add sample text (wrapped)
        import textwrap
        wrapped = textwrap.fill(sample, width=80)
        ax.text(0.05, y_pos, wrapped,
               fontsize=10, family='monospace', transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        y_pos -= 0.08 * (len(wrapped.split('\n')) + 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_learning_rate_comparison(lr_losses_dict, save_path=None):
    """
    Compare training curves for different learning rates.
    
    Args:
        lr_losses_dict: Dict mapping learning_rate -> list of losses
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    for lr, losses in sorted(lr_losses_dict.items()):
        # Smooth the curve
        if len(losses) > 100:
            window = 100
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), smoothed, 
                    linewidth=2, label=f'LR = {lr}', alpha=0.8)
        else:
            plt.plot(losses, linewidth=2, label=f'LR = {lr}', alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Rate Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_training_run(losses, samples_dict, save_dir=None):
    """
    Create a comprehensive visualization of a training run.
    
    Args:
        losses: List of training losses
        samples_dict: Dict mapping iteration -> generated sample
        save_dir: Directory to save all plots
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Loss curve
    save_path = os.path.join(save_dir, 'loss_curve.png') if save_dir else None
    plot_loss_curve(losses, save_path=save_path)
    
    # 2. Sample evolution
    fig, axes = plt.subplots(len(samples_dict), 1, figsize=(12, 3*len(samples_dict)))
    if len(samples_dict) == 1:
        axes = [axes]
    
    for ax, (iteration, sample) in zip(axes, sorted(samples_dict.items())):
        ax.axis('off')
        ax.text(0.05, 0.5, f"Iteration {iteration}:\n{sample[:200]}",
               fontsize=10, family='monospace', transform=ax.transAxes,
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sample_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved sample evolution to {save_dir}")
    else:
        plt.show()
    
    print("Visualization complete!")


if __name__ == "__main__":
    print("RNN Visualization Tools")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - plot_loss_curve")
    print("  - plot_hidden_states")
    print("  - plot_probability_distribution")
    print("  - plot_sequence_predictions")
    print("  - plot_gradient_flow")
    print("  - plot_temperature_comparison")
    print("  - plot_learning_rate_comparison")
    print("  - visualize_training_run")
    print("\nExample usage:")
    print("  from visualization import plot_loss_curve")
    print("  plot_loss_curve(losses, save_path='loss.png')")
