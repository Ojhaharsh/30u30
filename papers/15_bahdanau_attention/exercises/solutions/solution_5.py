"""
Solution 5: Visualize Attention Patterns

Beautiful visualizations that show what the model learned.
"""

import torch
import numpy as np
import sys
import os


def plot_attention_heatmap(attention_weights, source_tokens, target_tokens, 
                           title="Attention Weights", save_path=None):
    """
    Visualize attention weights as a heatmap.
    
    For the reversal task, we expect a reversed diagonal pattern:
    - Output position 0 attends to input position N-1
    - Output position 1 attends to input position N-2
    - etc.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed! Run: pip install matplotlib")
        return
    
    # Convert to numpy if tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, fontsize=11)
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens, fontsize=11)
    
    # Labels
    ax.set_xlabel('Source Sequence', fontsize=13)
    ax.set_ylabel('Target Sequence', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid for clarity
    ax.set_xticks(np.arange(-0.5, len(source_tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_tokens), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add attention values as text
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            value = attention_weights[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def plot_attention_over_time(attentions_list, source_tokens, 
                              title="Attention Evolution", save_path=None):
    """
    Show how attention evolves during decoding.
    
    Creates a row of bar charts, one for each decoding step.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed!")
        return
    
    n_steps = len(attentions_list)
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 4), sharey=True)
    
    if n_steps == 1:
        axes = [axes]
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(source_tokens)))
    
    for i, (ax, attn) in enumerate(zip(axes, attentions_list)):
        if torch.is_tensor(attn):
            attn = attn.cpu().numpy()
        
        bars = ax.bar(range(len(attn)), attn, color=colors)
        ax.set_title(f"Step {i+1}", fontsize=11)
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels(source_tokens, fontsize=9)
        ax.set_ylim(0, 1.1)
        
        # Highlight the maximum
        max_idx = np.argmax(attn)
        bars[max_idx].set_color('darkblue')
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(2)
        
        if i == 0:
            ax.set_ylabel('Attention Weight', fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def attention_entropy(attention_weights):
    """
    Calculate entropy of attention distribution.
    
    Low entropy = focused attention (the model knows where to look)
    High entropy = spread out attention (the model is confused)
    
    Perfect attention (all weight on one position): entropy ≈ 0
    Uniform attention: entropy = log(n)
    """
    if torch.is_tensor(attention_weights):
        eps = 1e-9
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps))
        return entropy.item()
    else:
        eps = 1e-9
        entropy = -np.sum(attention_weights * np.log(attention_weights + eps))
        return entropy


def check_diagonal_pattern(attention_weights, tolerance=1):
    """
    Check if attention follows reversed diagonal pattern.
    
    For the reversal task:
    - Output position i should attend to input position (n-1-i)
    - tolerance allows for some slack (nearby positions are okay)
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    target_len, source_len = attention_weights.shape
    correct = 0
    
    for i in range(target_len):
        peak = np.argmax(attention_weights[i])
        expected = source_len - 1 - i
        
        if abs(peak - expected) <= tolerance:
            correct += 1
    
    return correct / target_len


def analyze_attention_patterns(model, dataset, device, num_samples=5):
    """
    Analyze attention patterns across multiple examples.
    
    Returns detailed statistics about attention quality.
    """
    model.eval()
    
    entropies = []
    diagonal_accuracies = []
    examples = []
    
    for i in range(min(num_samples, len(dataset))):
        src, trg = dataset[i]
        src_t = src.unsqueeze(0).to(device)
        src_len = torch.tensor([len(src)]).to(device)
        
        pred, attentions = model.translate(src_t, src_len)
        
        # Get attention matrix (remove batch dimension)
        attn_matrix = attentions.squeeze(0).cpu()  # (trg_len, src_len)
        pred_tokens = pred.squeeze(0).cpu().tolist()
        
        # Remove EOS and after
        if 2 in pred_tokens:
            eos_pos = pred_tokens.index(2)
            pred_tokens = pred_tokens[:eos_pos]
            attn_matrix = attn_matrix[:eos_pos]
        
        # Compute metrics
        avg_entropy = np.mean([attention_entropy(attn_matrix[j]) 
                               for j in range(len(attn_matrix))])
        diag_acc = check_diagonal_pattern(attn_matrix)
        
        entropies.append(avg_entropy)
        diagonal_accuracies.append(diag_acc)
        
        examples.append({
            'source': src.tolist(),
            'target': trg[1:-1].tolist(),
            'prediction': pred_tokens,
            'attention': attn_matrix,
            'entropy': avg_entropy,
            'diagonal_accuracy': diag_acc,
            'correct': pred_tokens == trg[1:-1].tolist()
        })
    
    results = {
        'avg_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'avg_diagonal_accuracy': np.mean(diagonal_accuracies),
        'std_diagonal_accuracy': np.std(diagonal_accuracies),
        'examples': examples
    }
    
    return results


def visualize_model_attention(model, dataset, device, num_examples=3):
    """
    Create comprehensive visualization for a trained model.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required!")
        return
    
    model.eval()
    
    for i in range(min(num_examples, len(dataset))):
        src, trg = dataset[i]
        src_t = src.unsqueeze(0).to(device)
        src_len = torch.tensor([len(src)]).to(device)
        
        pred, attentions = model.translate(src_t, src_len)
        
        # Process outputs
        attn = attentions.squeeze(0).cpu()
        pred_list = pred.squeeze(0).cpu().tolist()
        
        if 2 in pred_list:
            eos_pos = pred_list.index(2)
            pred_list = pred_list[:eos_pos]
            attn = attn[:eos_pos]
        
        source_tokens = [str(t) for t in src.tolist()]
        target_tokens = [str(t) for t in pred_list]
        
        correct = pred_list == trg[1:-1].tolist()
        status = "✓ Correct" if correct else "✗ Wrong"
        
        print(f"\n{'='*50}")
        print(f"Example {i+1}: {status}")
        print(f"Input:  {src.tolist()}")
        print(f"Output: {pred_list}")
        print(f"Target: {trg[1:-1].tolist()}")
        
        # Plot attention
        plot_attention_heatmap(
            attn.numpy(), 
            source_tokens, 
            target_tokens,
            title=f"Example {i+1}: {status}",
            save_path=f"attention_example_{i+1}.png"
        )


# ============================================================================
# Demo (works without training)
# ============================================================================

def demo_visualization():
    """Create demo visualizations with synthetic data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install matplotlib: pip install matplotlib")
        return
    
    print("Creating demo attention visualizations...")
    
    # Simulate perfect reversal attention
    seq_len = 6
    source = ['5', '3', '8', '2', '1', '7']
    target = ['7', '1', '2', '8', '3', '5']
    
    # Create perfect reversed diagonal
    attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        attention[i, seq_len - 1 - i] = 0.85
        # Add some realistic noise
        for j in range(seq_len):
            if j != seq_len - 1 - i:
                attention[i, j] = 0.15 / (seq_len - 1)
    
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    # Plot heatmap
    print("\n1. Attention Heatmap")
    plot_attention_heatmap(attention, source, target,
                          title="Reversal Task: Perfect Attention Pattern",
                          save_path="demo_heatmap.png")
    
    # Plot attention over time
    print("\n2. Attention Evolution")
    attn_steps = [attention[i] for i in range(seq_len)]
    plot_attention_over_time(attn_steps, source,
                            title="How Attention Shifts During Decoding",
                            save_path="demo_evolution.png")
    
    # Analysis
    print("\n3. Attention Analysis")
    print("="*50)
    
    for i in range(seq_len):
        ent = attention_entropy(attention[i])
        peak = np.argmax(attention[i])
        expected = seq_len - 1 - i
        correct = "✓" if peak == expected else "✗"
        print(f"  Step {i+1}: Output '{target[i]}' → Input '{source[peak]}' "
              f"(entropy: {ent:.3f}) {correct}")
    
    diag_acc = check_diagonal_pattern(attention)
    print(f"\n  Diagonal Accuracy: {diag_acc:.0%}")
    print(f"  Average Entropy: {np.mean([attention_entropy(attention[i]) for i in range(seq_len)]):.3f}")
    print("\n  Low entropy = focused attention (good!)")
    print("  High diagonal accuracy = correct pattern learned!")


def main():
    """Run demo or trained model visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Visualization")
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    if args.demo or not args.model:
        demo_visualization()
    else:
        print("To visualize a trained model, first train with solution_4.py")
        print("then call visualize_model_attention(model, dataset, device)")
        demo_visualization()


if __name__ == '__main__':
    main()
