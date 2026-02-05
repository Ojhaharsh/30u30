"""
Solution to Exercise 2: Gate Analysis
=====================================

Complete implementation for analyzing LSTM gate behaviors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from lstm_implementation import LSTM


def capture_gate_activations(lstm, text, char_to_idx):
    """
    Process text through LSTM and capture all gate activations.
    
    Returns:
        Dict with keys: 'forget', 'input', 'cell_candidate', 'output', 'cell', 'hidden'
        Each contains a list of activations over time.
    """
    # Initialize states
    h = np.zeros(lstm.hidden_size)
    c = np.zeros(lstm.hidden_size)
    
    # Storage for activations
    activations = {
        'forget': [],
        'input': [],
        'cell_candidate': [],
        'output': [],
        'cell': [],
        'hidden': []
    }
    
    # Process each character
    for char in text:
        if char not in char_to_idx:
            continue
        
        idx = char_to_idx[char]
        x = np.zeros(lstm.vocab_size)
        x[idx] = 1
        
        # Concatenate input and hidden state
        combined = np.concatenate([h, x])
        
        # Compute gates
        f = sigmoid(np.dot(lstm.Wf, combined) + lstm.bf)  # Forget gate
        i = sigmoid(np.dot(lstm.Wi, combined) + lstm.bi)  # Input gate
        c_tilde = np.tanh(np.dot(lstm.Wc, combined) + lstm.bc)  # Cell candidate
        c = f * c + i * c_tilde  # New cell state
        o = sigmoid(np.dot(lstm.Wo, combined) + lstm.bo)  # Output gate
        h = o * np.tanh(c)  # New hidden state
        
        # Store activations (average across hidden dimensions)
        activations['forget'].append(np.mean(f))
        activations['input'].append(np.mean(i))
        activations['cell_candidate'].append(np.mean(c_tilde))
        activations['output'].append(np.mean(o))
        activations['cell'].append(np.mean(c))
        activations['hidden'].append(np.mean(h))
    
    return activations


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def analyze_gate_patterns(activations, text):
    """
    Analyze patterns in gate activations.
    
    Insights to extract:
    1. When does forget gate activate? (What does it forget?)
    2. When does input gate activate? (What does it store?)
    3. How do gates coordinate?
    4. Relationship to input text structure
    """
    print("="*70)
    print("GATE ACTIVATION ANALYSIS")
    print("="*70)
    
    # 1. Basic statistics
    print("\n[1] GATE STATISTICS")
    print("-" * 70)
    for gate_name in ['forget', 'input', 'output']:
        values = activations[gate_name]
        print(f"\n{gate_name.upper()} GATE:")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Std:  {np.std(values):.3f}")
        print(f"  Min:  {np.min(values):.3f}")
        print(f"  Max:  {np.max(values):.3f}")
    
    # 2. Correlation analysis
    print("\n[2] GATE CORRELATIONS")
    print("-" * 70)
    
    forget = np.array(activations['forget'])
    input_gate = np.array(activations['input'])
    output = np.array(activations['output'])
    
    # Forget vs Input (should be negatively correlated)
    corr_fi = np.corrcoef(forget, input_gate)[0, 1]
    print(f"Forget ↔ Input:  {corr_fi:+.3f}")
    if corr_fi < -0.3:
        print("  ✓ Good! Gates are complementary (forget old when inputting new)")
    else:
        print("  Unexpected: Gates should be more anti-correlated")
    
    # Input vs Output
    corr_io = np.corrcoef(input_gate, output)[0, 1]
    print(f"Input ↔ Output:  {corr_io:+.3f}")
    
    # 3. Identify key moments
    print("\n[3] KEY MOMENTS")
    print("-" * 70)
    
    # When forget gate is very active (>0.8)
    high_forget = np.where(forget > 0.8)[0]
    if len(high_forget) > 0:
        print(f"\nHigh FORGET activity at {len(high_forget)} positions:")
        for idx in high_forget[:5]:  # Show first 5
            if idx < len(text):
                context = text[max(0, idx-10):idx+10]
                print(f"  Position {idx}: ...{context}...")
    
    # When input gate is very active (>0.8)
    high_input = np.where(input_gate > 0.8)[0]
    if len(high_input) > 0:
        print(f"\nHigh INPUT activity at {len(high_input)} positions:")
        for idx in high_input[:5]:
            if idx < len(text):
                context = text[max(0, idx-10):idx+10]
                print(f"  Position {idx}: ...{context}...")
    
    # 4. Structural patterns
    print("\n[4] STRUCTURAL PATTERNS")
    print("-" * 70)
    
    # Check activity at sentence boundaries
    sentence_ends = [i for i, char in enumerate(text) if char in '.!?']
    
    if sentence_ends:
        avg_forget_at_end = np.mean([forget[i] for i in sentence_ends if i < len(forget)])
        avg_forget_overall = np.mean(forget)
        
        print(f"\nForget gate at sentence ends: {avg_forget_at_end:.3f}")
        print(f"Forget gate overall:          {avg_forget_overall:.3f}")
        
        if avg_forget_at_end > avg_forget_overall + 0.1:
            print("  ✓ LSTM forgets more at sentence boundaries!")
        else:
            print("  → LSTM doesn't show strong forgetting at sentence ends")


def visualize_gates(activations, text, save_path='gate_analysis.png'):
    """
    Create comprehensive 4-panel visualization of gate activations.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    time_steps = range(len(activations['forget']))
    
    # Panel 1: All three gates
    axes[0].plot(time_steps, activations['forget'], label='Forget gate', 
                color='red', alpha=0.7, linewidth=1.5)
    axes[0].plot(time_steps, activations['input'], label='Input gate', 
                color='green', alpha=0.7, linewidth=1.5)
    axes[0].plot(time_steps, activations['output'], label='Output gate', 
                color='blue', alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel('Gate Activation', fontsize=12)
    axes[0].set_title('Gate Activations Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Panel 2: Cell state evolution
    axes[1].plot(time_steps, activations['cell'], color='purple', linewidth=2)
    axes[1].set_ylabel('Cell State', fontsize=12)
    axes[1].set_title('Cell State Evolution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Panel 3: Gate complementarity (Forget vs Input)
    axes[2].plot(time_steps, activations['forget'], label='Forget', 
                color='red', alpha=0.6, linewidth=1.5)
    axes[2].plot(time_steps, activations['input'], label='Input', 
                color='green', alpha=0.6, linewidth=1.5)
    axes[2].plot(time_steps, 
                np.array(activations['forget']) + np.array(activations['input']),
                label='Forget + Input', color='black', linestyle='--', linewidth=2)
    axes[2].set_ylabel('Activation', fontsize=12)
    axes[2].set_title('Gate Complementarity (Forget vs Input)', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 2])
    
    # Panel 4: Hidden state activity
    axes[3].plot(time_steps, activations['hidden'], color='orange', linewidth=2)
    axes[3].set_ylabel('Hidden State', fontsize=12)
    axes[3].set_xlabel('Time Step', fontsize=12)
    axes[3].set_title('Hidden State Activity', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    return fig


def compare_gates_across_contexts(lstm, char_to_idx):
    """
    Compare gate behaviors in different contexts.
    
    Test cases:
    1. Repetitive text: "the the the the"
    2. Long-range dependency: "The dog, which was big, barked."
    3. Changing context: "It was sunny. It was rainy."
    """
    test_cases = {
        'Repetitive': "the the the the the the",
        'Long-range dependency': "The cat that was sleeping on the mat stood up",
        'Context switch': "It was hot. Now it is cold."
    }
    
    print("\n" + "="*70)
    print("GATE BEHAVIOR ACROSS CONTEXTS")
    print("="*70)
    
    for name, text in test_cases.items():
        print(f"\n{name.upper()}: '{text}'")
        print("-" * 70)
        
        activations = capture_gate_activations(lstm, text, char_to_idx)
        
        forget_vals = activations['forget']
        input_vals = activations['input']
        
        print(f"Forget gate - Mean: {np.mean(forget_vals):.3f}, Std: {np.std(forget_vals):.3f}")
        print(f"Input gate  - Mean: {np.mean(input_vals):.3f}, Std: {np.std(input_vals):.3f}")
        
        # Analysis
        if 'Repetitive' in name:
            if np.mean(forget_vals) > 0.7:
                print("  → High forget: LSTM discards repetitive information ✓")
        
        elif 'Long-range' in name:
            if np.std(input_vals) > 0.1:
                print("  → Variable input: LSTM selectively stores information ✓")
        
        elif 'Context switch' in name:
            mid = len(forget_vals) // 2
            early_forget = np.mean(forget_vals[:mid])
            late_forget = np.mean(forget_vals[mid:])
            if late_forget > early_forget + 0.1:
                print("  → Higher forgetting after context switch ✓")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("LSTM GATE ANALYSIS TOOLKIT")
    print("="*70)
    
    print("\nThis solution provides:")
    print("  1. Gate activation capture during forward pass")
    print("  2. Statistical analysis of gate behaviors")
    print("  3. Correlation analysis between gates")
    print("  4. Identification of key moments (high activity)")
    print("  5. Structural pattern detection (sentence boundaries)")
    print("  6. 4-panel visualization")
    print("  7. Cross-context comparison")
    
    print("\n" + "-"*70)
    print("KEY INSIGHTS FROM GATE ANALYSIS")
    print("-"*70)
    
    print("\n1. FORGET GATE:")
    print("   • Activates when old information is no longer relevant")
    print("   • High at sentence boundaries")
    print("   • High during context switches")
    print("   • Negatively correlated with input gate")
    
    print("\n2. INPUT GATE:")
    print("   • Activates when new important information appears")
    print("   • High at content words (nouns, verbs)")
    print("   • Low at function words (the, a, of)")
    print("   • Complementary to forget gate")
    
    print("\n3. OUTPUT GATE:")
    print("   • Controls what information to expose")
    print("   • High when prediction is confident")
    print("   • Regulates hidden state magnitude")
    
    print("\n4. CELL STATE:")
    print("   • Accumulates information over time")
    print("   • Stable in meaningful contexts")
    print("   • Resets at topic boundaries")
    
    print("\n" + "="*70)
    print("Run this on your trained LSTM to see gate behaviors!")
    print("="*70)
