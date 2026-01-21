"""
Visualization Tools for LSTM Networks
=====================================

Functions to visualize and understand LSTM behavior:
1. Gate activation patterns (forget, input, output)
2. Cell state evolution over time
3. Gradient flow comparison (LSTM vs vanilla RNN)
4. Long-range dependency tests
5. Temperature effects on sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gate_activations(gates_dict, sequence_text, save_path=None):
    """
    Visualize forget, input, and output gate activations over a sequence.
    
    This shows WHEN the LSTM decides to remember, forget, or output information.
    
    Args:
        gates_dict: Dict with keys 'forget', 'input', 'output', each containing
                   arrays of shape (seq_len, hidden_size)
        sequence_text: String of characters corresponding to time steps
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    gates = ['forget', 'input', 'output']
    titles = ['Forget Gate (1=keep, 0=forget)', 
              'Input Gate (1=add, 0=ignore)', 
              'Output Gate (1=output, 0=hide)']
    
    for ax, gate_name, title in zip(axes, gates, titles):
        gate_data = np.array(gates_dict[gate_name]).T  # (hidden_size, seq_len)
        
        # Plot heatmap
        im = ax.imshow(gate_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xlabel('Time Step (Character)', fontsize=12)
        ax.set_ylabel('Hidden Unit', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Show characters on x-axis (if not too many)
        if len(sequence_text) <= 50:
            ax.set_xticks(range(len(sequence_text)))
            ax.set_xticklabels(list(sequence_text), fontsize=8)
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Gate Activation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_cell_state_evolution(cell_states, sequence_text, save_path=None):
    """
    Visualize how cell state evolves over a sequence.
    
    The cell state is the "memory" of the LSTM. This shows what information
    is being preserved across time steps.
    
    Args:
        cell_states: Array of shape (seq_len, hidden_size)
        sequence_text: String of characters
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(14, 7))
    
    # Transpose so rows are hidden units, columns are time steps
    cell_states_T = np.array(cell_states).T  # (hidden_size, seq_len)
    
    # Plot heatmap
    im = plt.imshow(cell_states_T, cmap='RdBu_r', aspect='auto', 
                    vmin=-1, vmax=1, interpolation='nearest')
    
    plt.xlabel('Time Step (Character)', fontsize=12)
    plt.ylabel('Hidden Unit', fontsize=12)
    plt.title('LSTM Cell State Evolution\n(Red=positive memory, Blue=negative memory)', 
             fontsize=14, fontweight='bold')
    
    # Show characters on x-axis (if not too many)
    if len(sequence_text) <= 50:
        plt.xticks(range(len(sequence_text)), list(sequence_text), fontsize=9)
    
    plt.colorbar(im, label='Cell State Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_gradient_flow_comparison(lstm_grads, rnn_grads=None, save_path=None):
    """
    Compare gradient flow in LSTM vs vanilla RNN.
    
    This demonstrates why LSTMs solve the vanishing gradient problem!
    
    Args:
        lstm_grads: List of gradient norms at each time step (LSTM)
        rnn_grads: List of gradient norms for vanilla RNN (optional)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    time_steps = range(len(lstm_grads))
    
    # Plot LSTM gradients
    plt.plot(time_steps, lstm_grads, marker='o', linewidth=2, 
            label='LSTM', color='green', markersize=4, alpha=0.8)
    
    # Plot RNN gradients if provided
    if rnn_grads is not None:
        plt.plot(time_steps, rnn_grads, marker='s', linewidth=2, 
                label='Vanilla RNN', color='red', markersize=4, alpha=0.8)
    
    # Reference lines
    plt.axhline(y=1.0, color='gray', linestyle='--', 
               label='Gradient norm = 1', alpha=0.5)
    plt.axhline(y=0.1, color='orange', linestyle='--', 
               label='Vanishing threshold (0.1)', alpha=0.5)
    
    plt.xlabel('Time Step (backward)', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title('Gradient Flow Through Time: LSTM vs RNN', 
             fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def analyze_gate_patterns(gates_dict, sequence_text):
    """
    Print statistical analysis of gate activation patterns.
    
    Args:
        gates_dict: Dict with 'forget', 'input', 'output' gates
        sequence_text: String of characters
    """
    print("=" * 60)
    print("GATE ACTIVATION ANALYSIS")
    print("=" * 60)
    
    for gate_name in ['forget', 'input', 'output']:
        gate_data = np.array(gates_dict[gate_name])  # (seq_len, hidden_size)
        
        print(f"\n{gate_name.upper()} GATE:")
        print(f"  Mean activation: {gate_data.mean():.3f}")
        print(f"  Std deviation: {gate_data.std():.3f}")
        print(f"  % Fully open (>0.9): {(gate_data > 0.9).mean() * 100:.1f}%")
        print(f"  % Fully closed (<0.1): {(gate_data < 0.1).mean() * 100:.1f}%")
        print(f"  % Uncertain (0.4-0.6): {((gate_data > 0.4) & (gate_data < 0.6)).mean() * 100:.1f}%")
    
    # Analyze specific positions
    print("\n" + "=" * 60)
    print("CHARACTER-BY-CHARACTER ANALYSIS")
    print("=" * 60)
    
    forget_data = np.array(gates_dict['forget'])
    input_data = np.array(gates_dict['input'])
    output_data = np.array(gates_dict['output'])
    
    print(f"\n{'Char':<6} | {'Forget':<8} | {'Input':<8} | {'Output':<8} | {'Interpretation'}")
    print("-" * 70)
    
    for t, char in enumerate(sequence_text[:20]):  # First 20 characters
        f_mean = forget_data[t].mean()
        i_mean = input_data[t].mean()
        o_mean = output_data[t].mean()
        
        # Interpret
        if f_mean > 0.7 and i_mean < 0.3:
            interp = "Keeping old info"
        elif f_mean < 0.3 and i_mean > 0.7:
            interp = "Replacing memory"
        elif f_mean > 0.7 and i_mean > 0.7:
            interp = "Accumulating info"
        elif o_mean > 0.7:
            interp = "Outputting strongly"
        else:
            interp = "Mixed signals"
        
        char_display = char if char not in ['\n', ' '] else ('\\n' if char == '\n' else '_')
        print(f"{char_display:<6} | {f_mean:>6.3f}   | {i_mean:>6.3f}   | {o_mean:>6.3f}   | {interp}")


def plot_long_range_dependency_test(distances, accuracies, save_path=None):
    """
    Plot accuracy vs distance for long-range dependency task.
    
    This tests: "Can the LSTM remember the first character after N steps?"
    
    Args:
        distances: List of sequence lengths tested
        accuracies: Accuracy at each distance
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(distances, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    
    plt.xlabel('Distance (number of intervening characters)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Long-Range Dependency Test\n"Remember first character after N steps"', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add annotations
    if len(distances) > 0:
        best_idx = np.argmax(accuracies)
        plt.annotate(f'Best: {accuracies[best_idx]:.1f}% at {distances[best_idx]} steps',
                    xy=(distances[best_idx], accuracies[best_idx]),
                    xytext=(distances[best_idx] + 5, accuracies[best_idx] - 10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_training_comparison(lstm_losses, rnn_losses=None, save_path=None):
    """
    Compare training curves for LSTM vs vanilla RNN.
    
    Args:
        lstm_losses: List of losses for LSTM
        rnn_losses: List of losses for RNN (optional)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Smooth curves
    window = min(100, len(lstm_losses) // 10)
    if window > 1:
        lstm_smoothed = np.convolve(lstm_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(lstm_losses)), lstm_smoothed, 
                linewidth=2, label='LSTM', color='green')
    else:
        plt.plot(lstm_losses, linewidth=2, label='LSTM', color='green')
    
    if rnn_losses is not None:
        if window > 1:
            rnn_smoothed = np.convolve(rnn_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rnn_losses)), rnn_smoothed, 
                    linewidth=2, label='Vanilla RNN', color='red', alpha=0.7)
        else:
            plt.plot(rnn_losses, linewidth=2, label='Vanilla RNN', color='red', alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss: LSTM vs Vanilla RNN', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_attention_to_memory(cell_states, query_indices, sequence_text, save_path=None):
    """
    Visualize which parts of cell state are "active" at specific time steps.
    
    This helps understand what information the LSTM is focusing on.
    
    Args:
        cell_states: Array of shape (seq_len, hidden_size)
        query_indices: List of time steps to highlight
        sequence_text: String of characters
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(len(query_indices), 1, 
                             figsize=(12, 3*len(query_indices)))
    
    if len(query_indices) == 1:
        axes = [axes]
    
    for ax, t in zip(axes, query_indices):
        # Get cell state at this time step
        C_t = cell_states[t]  # (hidden_size,)
        
        # Plot as bar chart
        ax.bar(range(len(C_t)), C_t, color=['red' if x < 0 else 'blue' for x in C_t])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        char = sequence_text[t] if t < len(sequence_text) else '?'
        char_display = char if char not in ['\n', ' '] else ('\\n' if char == '\n' else 'SPACE')
        
        ax.set_xlabel('Hidden Unit', fontsize=10)
        ax.set_ylabel('Cell State Value', fontsize=10)
        ax.set_title(f"Cell State at t={t} (char: '{char_display}')", 
                    fontsize=12, fontweight='bold')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def create_comprehensive_report(lstm_data, save_dir='lstm_analysis'):
    """
    Generate a comprehensive analysis report with multiple visualizations.
    
    Args:
        lstm_data: Dict containing:
            - 'gates': Dict of gate activations
            - 'cell_states': Array of cell states
            - 'sequence': Text sequence
            - 'losses': Training losses
            - (optional) 'rnn_losses': RNN losses for comparison
            - (optional) 'gradient_norms': Gradient norms
        save_dir: Directory to save all plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating comprehensive LSTM analysis in {save_dir}/...")
    
    # 1. Gate activations
    if 'gates' in lstm_data and 'sequence' in lstm_data:
        print("  - Plotting gate activations...")
        plot_gate_activations(lstm_data['gates'], lstm_data['sequence'],
                             save_path=f"{save_dir}/gate_activations.png")
    
    # 2. Cell state evolution
    if 'cell_states' in lstm_data and 'sequence' in lstm_data:
        print("  - Plotting cell state evolution...")
        plot_cell_state_evolution(lstm_data['cell_states'], lstm_data['sequence'],
                                  save_path=f"{save_dir}/cell_state_evolution.png")
    
    # 3. Training curves
    if 'losses' in lstm_data:
        print("  - Plotting training curves...")
        rnn_losses = lstm_data.get('rnn_losses', None)
        plot_training_comparison(lstm_data['losses'], rnn_losses,
                                save_path=f"{save_dir}/training_comparison.png")
    
    # 4. Gradient flow
    if 'gradient_norms' in lstm_data:
        print("  - Plotting gradient flow...")
        rnn_grads = lstm_data.get('rnn_gradient_norms', None)
        plot_gradient_flow_comparison(lstm_data['gradient_norms'], rnn_grads,
                                     save_path=f"{save_dir}/gradient_flow.png")
    
    # 5. Gate analysis
    if 'gates' in lstm_data and 'sequence' in lstm_data:
        print("  - Analyzing gate patterns...")
        with open(f"{save_dir}/gate_analysis.txt", 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            analyze_gate_patterns(lstm_data['gates'], lstm_data['sequence'])
            sys.stdout = old_stdout
    
    print(f"\nAnalysis complete! Results saved to {save_dir}/")
    print(f"  - gate_activations.png")
    print(f"  - cell_state_evolution.png")
    print(f"  - training_comparison.png")
    print(f"  - gradient_flow.png (if data provided)")
    print(f"  - gate_analysis.txt")


if __name__ == "__main__":
    print("LSTM Visualization Tools")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - plot_gate_activations()")
    print("  - plot_cell_state_evolution()")
    print("  - plot_gradient_flow_comparison()")
    print("  - analyze_gate_patterns()")
    print("  - plot_long_range_dependency_test()")
    print("  - plot_training_comparison()")
    print("  - visualize_attention_to_memory()")
    print("  - create_comprehensive_report()")
    print("\nExample usage:")
    print("  from visualization import plot_gate_activations")
    print("  plot_gate_activations(gates, text, save_path='gates.png')")
