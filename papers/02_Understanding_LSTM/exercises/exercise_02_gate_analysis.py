"""
Exercise 2: Gate Activation Analysis
=====================================

Goal: Train an LSTM and analyze what each gate learns to do.

Your Task:
- Train an LSTM on a simple repeating pattern
- Capture and visualize gate activations
- Analyze when gates open/close
- Write a report on your findings

Learning Objectives:
1. Understand when forget gates activate (what to throw away)
2. See when input gates activate (what to add)
3. Observe when output gates activate (what to reveal)
4. Learn how gates coordinate to solve tasks

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM
from visualization import plot_gate_activations, analyze_gate_patterns


def create_repeating_pattern(pattern="abc", repeats=10):
    """
    Create a simple repeating pattern for analysis.
    
    Args:
        pattern: String pattern to repeat (e.g., "abc")
        repeats: Number of times to repeat
    
    Returns:
        text: Repeated pattern string
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    # TODO 1: Create the text by repeating the pattern
    # Hint: text = pattern * repeats
    text = None  # Replace with your code
    
    # TODO 2: Create vocabulary mappings
    chars = sorted(list(set(text)))
    char_to_idx = {}  # Fill this in
    idx_to_char = {}  # Fill this in
    
    return text, char_to_idx, idx_to_char


def train_and_capture_gates(text, char_to_idx, hidden_size=32, num_iterations=500):
    """
    Train LSTM and capture gate activations.
    
    Args:
        text: Training text
        char_to_idx: Character to index mapping
        hidden_size: Size of hidden layer
        num_iterations: Number of training iterations
    
    Returns:
        lstm: Trained LSTM model
        gates_dict: Dictionary of gate activations
        cell_states: List of cell states
    """
    vocab_size = len(char_to_idx)
    
    # TODO 3: Create LSTM
    lstm = None  # Replace with LSTM initialization
    
    # TODO 4: Training loop
    # Initialize states
    h_prev = np.zeros(hidden_size)
    C_prev = np.zeros(hidden_size)
    
    seq_length = 10
    learning_rate = 0.01
    
    print("Training LSTM...")
    for iteration in range(num_iterations):
        # TODO 5: Sample a random sequence from text
        # Hint: start = np.random.randint(0, len(text) - seq_length - 1)
        pass
        
        # TODO 6: Forward pass
        
        # TODO 7: Backward pass
        
        # TODO 8: Update weights
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}")
    
    # TODO 9: Capture gate activations on a test sequence
    # Run forward pass and manually capture f_t, i_t, o_t for each step
    test_seq = text[:20]
    gates_dict = {
        'forget': [],
        'input': [],
        'output': []
    }
    cell_states = []
    
    # TODO 10: Fill gates_dict and cell_states by running forward pass
    # and manually computing gates (see implementation.py for reference)
    
    return lstm, gates_dict, cell_states, test_seq


def analyze_gates(gates_dict, cell_states, sequence):
    """
    Analyze and visualize gate patterns.
    
    Args:
        gates_dict: Dictionary with 'forget', 'input', 'output' keys
        cell_states: List of cell state arrays
        sequence: Text sequence corresponding to time steps
    """
    print("\n" + "=" * 60)
    print("GATE ACTIVATION ANALYSIS")
    print("=" * 60)
    
    # TODO 11: Use visualization functions
    # plot_gate_activations(gates_dict, sequence)
    
    # TODO 12: Compute statistics
    # For each gate, compute:
    # - Mean activation
    # - Std deviation
    # - % fully open (>0.9)
    # - % fully closed (<0.1)
    
    print("\nYour analysis here...")


def write_report(gates_dict, sequence):
    """
    Write a report on your findings.
    
    TODO 13: Answer these questions in your report:
    
    1. Which gates are most active on average?
    2. Do any gates specialize (always on or off)?
    3. How do gates respond to specific characters?
    4. Do you see any patterns related to the repeating structure?
    5. What surprised you most?
    """
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print("\nWrite your findings here...")


if __name__ == "__main__":
    print(__doc__)
    
    # Create simple pattern
    text, char_to_idx, idx_to_char = create_repeating_pattern("abc", repeats=20)
    print(f"Pattern: {text[:30]}...")
    
    # Train and capture gates
    lstm, gates_dict, cell_states, sequence = train_and_capture_gates(
        text, char_to_idx
    )
    
    # Analyze
    analyze_gates(gates_dict, cell_states, sequence)
    
    # Write report
    write_report(gates_dict, sequence)
    
    print("\n✅ Exercise 2 complete!")
    print("Compare your results with solutions/solution_02_gate_analysis.py")
