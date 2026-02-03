"""
Exercise 5: Visualize Attention Patterns

"Show me what the model is looking at."

Attention visualization is one of the most beautiful aspects of this paper.
For a reversal task, we expect to see a reversed diagonal pattern:

    Output position 1 → Input position N (last)
    Output position 2 → Input position N-1
    ...
    Output position N → Input position 1 (first)

Your task: Create beautiful attention heatmaps that show what the model learned!
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_attention_heatmap(attention_weights, source_tokens, target_tokens, 
                           title="Attention Weights", save_path=None):
    """
    TODO: Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: 2D tensor/array of shape (target_len, source_len)
        source_tokens: List of source tokens (strings or ints)
        target_tokens: List of target tokens (strings or ints)
        title: Plot title
        save_path: If provided, save figure to this path
    
    Steps:
    1. Import matplotlib.pyplot as plt
    2. Create figure with plt.figure(figsize=(8, 6))
    3. Use plt.imshow() with cmap='Blues' or 'viridis'
    4. Add colorbar with plt.colorbar()
    5. Set x-axis labels to source tokens
    6. Set y-axis labels to target tokens
    7. Add title
    8. Label axes: "Source" and "Target"
    9. Save or show
    
    Hint: Use plt.xticks(range(len(source_tokens)), source_tokens)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed! Run: pip install matplotlib")
        return
    
    # Convert to numpy if tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    # TODO: Create the heatmap visualization
    # 
    # plt.figure(figsize=(8, 6))
    # plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    # plt.colorbar(label='Attention Weight')
    # ...
    
    raise NotImplementedError("Implement plot_attention_heatmap!")


def plot_attention_over_time(attentions_list, source_tokens, 
                              title="Attention Evolution", save_path=None):
    """
    TODO: Show how attention evolves during decoding.
    
    Create a grid of subplots, one for each decoding step.
    Each subplot shows where the model is looking at that step.
    
    Args:
        attentions_list: List of 1D attention vectors, one per decoding step
        source_tokens: List of source tokens
        title: Overall title
        save_path: If provided, save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed!")
        return
    
    # TODO: Create a row of bar plots showing attention at each step
    #
    # n_steps = len(attentions_list)
    # fig, axes = plt.subplots(1, n_steps, figsize=(3*n_steps, 3))
    # 
    # for i, (ax, attn) in enumerate(zip(axes, attentions_list)):
    #     ax.bar(range(len(attn)), attn)
    #     ax.set_title(f"Step {i+1}")
    #     ax.set_xticks(range(len(source_tokens)))
    #     ax.set_xticklabels(source_tokens)
    #     ax.set_ylim(0, 1)
    # ...
    
    raise NotImplementedError("Implement plot_attention_over_time!")


def analyze_attention_patterns(model, dataset, device, num_samples=5):
    """
    TODO: Analyze attention patterns across multiple examples.
    
    For the reversal task, check:
    1. Does attention form a reversed diagonal pattern?
    2. How sharp/focused is the attention at each step?
    3. Are there any "confused" positions?
    
    Args:
        model: Trained Seq2Seq model
        dataset: Dataset to sample from
        device: torch device
        num_samples: Number of examples to analyze
    
    Returns:
        dict with analysis results:
        - 'avg_entropy': Average entropy of attention distributions
        - 'diagonal_accuracy': How often attention peaks are on reversed diagonal
        - 'examples': List of (source, target, attention, prediction) tuples
    """
    model.eval()
    
    results = {
        'avg_entropy': 0.0,
        'diagonal_accuracy': 0.0,
        'examples': []
    }
    
    # TODO: Implement attention analysis
    #
    # for i in range(num_samples):
    #     src, trg = dataset[i]
    #     src_t = src.unsqueeze(0).to(device)
    #     src_len = torch.tensor([len(src)]).to(device)
    #     
    #     pred, attentions = model.translate(src_t, src_len)
    #     
    #     # Analyze attention patterns...
    #     # - Compute entropy: -sum(a * log(a + 1e-9))
    #     # - Check if argmax is on reversed diagonal
    #     
    #     results['examples'].append((
    #         src.tolist(),
    #         trg[1:-1].tolist(),
    #         attentions.squeeze().cpu(),
    #         pred.squeeze().cpu().tolist()
    #     ))
    
    raise NotImplementedError("Implement analyze_attention_patterns!")


def attention_entropy(attention_weights):
    """
    Calculate entropy of attention distribution.
    
    Low entropy = focused attention (good!)
    High entropy = spread out attention (unfocused)
    
    Args:
        attention_weights: Tensor of attention weights
        
    Returns:
        Entropy value (scalar)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps))
    return entropy.item()


def check_diagonal_pattern(attention_weights, tolerance=1):
    """
    Check if attention follows reversed diagonal pattern.
    
    For reversal task, output position i should attend to input position (n-1-i).
    
    Args:
        attention_weights: 2D tensor (target_len, source_len)
        tolerance: How many positions off is still considered correct
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    target_len, source_len = attention_weights.shape
    correct = 0
    
    for i in range(target_len):
        peak = attention_weights[i].argmax().item()
        expected = source_len - 1 - i
        
        if abs(peak - expected) <= tolerance:
            correct += 1
    
    return correct / target_len


# ============================================================================
# Demo Visualization (works without training)
# ============================================================================

def demo_visualization():
    """Create a demo visualization with synthetic attention."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install matplotlib: pip install matplotlib")
        return
    
    print("Creating demo attention visualization...")
    
    # Simulate perfect reversal attention (reversed diagonal)
    seq_len = 6
    attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        # Perfect reversal: position i attends to position (n-1-i)
        attention[i, seq_len - 1 - i] = 0.8
        # Some noise
        if i > 0:
            attention[i, seq_len - i] = 0.1
        if i < seq_len - 1:
            attention[i, seq_len - 2 - i] = 0.1
    
    # Normalize rows
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    source = ['5', '3', '8', '2', '1', '7']
    target = ['7', '1', '2', '8', '3', '5']
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(attention, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    plt.xticks(range(len(source)), source, fontsize=12)
    plt.yticks(range(len(target)), target, fontsize=12)
    
    plt.xlabel('Source Sequence', fontsize=12)
    plt.ylabel('Target Sequence', fontsize=12)
    plt.title('Attention Pattern for Reversal Task\n(Reversed Diagonal = Correct!)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('attention_demo.png', dpi=150)
    print("Saved to attention_demo.png")
    plt.show()
    
    # Analysis
    print("\nAttention Analysis:")
    print(f"  Pattern forms reversed diagonal: ✓")
    print(f"  Each output token focuses on correct input")
    print(f"  This is what a well-trained model looks like!")


def test_implementation():
    """Test your implementations."""
    print("Testing implementations...")
    
    # Test data
    attention = torch.tensor([
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1]
    ])
    source = ['A', 'B', 'C']
    target = ['C', 'B', 'A']
    
    # Test entropy
    print("\nTesting attention_entropy...")
    ent = attention_entropy(attention[0])
    print(f"  Entropy of focused attention: {ent:.3f}")
    
    uniform = torch.ones(3) / 3
    ent_uniform = attention_entropy(uniform)
    print(f"  Entropy of uniform attention: {ent_uniform:.3f}")
    print(f"  (Lower entropy = more focused)")
    
    # Test diagonal check
    print("\nTesting check_diagonal_pattern...")
    acc = check_diagonal_pattern(attention)
    print(f"  Diagonal accuracy: {acc:.0%}")
    
    # Test visualization (if matplotlib available)
    try:
        print("\nTesting plot_attention_heatmap...")
        plot_attention_heatmap(attention, source, target, 
                               title="Test Attention", save_path="test_attention.png")
        print("  ✓ Visualization created!")
    except NotImplementedError:
        print("  (Not implemented yet)")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*50)
    print("Implement the TODO functions to complete this exercise!")
    print("Then run demo_visualization() to see a beautiful heatmap.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo visualization')
    parser.add_argument('--test', action='store_true', help='Test implementations')
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    elif args.test:
        test_implementation()
    else:
        print("Attention Visualization Exercise")
        print("=" * 40)
        print("\nOptions:")
        print("  --demo  : See demo visualization (no training needed)")
        print("  --test  : Test your implementations")
        print("\nStart by implementing the TODO functions, then run with --test")
        print("\nThis exercise teaches you to:")
        print("  1. Visualize what the model is 'looking at'")
        print("  2. Understand attention patterns")
        print("  3. Debug and interpret model behavior")
