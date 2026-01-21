"""
Exercise 5: LSTM vs GRU Comparison
===================================

Goal: Implement GRU and compare it to LSTM.

Your Task:
- Implement a GRU (Gated Recurrent Unit) from scratch
- Train both LSTM and GRU on the same task
- Compare: speed, performance, parameters
- Write a comparison report

Learning Objectives:
1. Understand GRU architecture (3 gates vs 4)
2. See trade-offs between complexity and performance
3. Learn when to use GRU vs LSTM
4. Appreciate design simplifications

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM


class GRU:
    """
    Gated Recurrent Unit (GRU) implementation.
    
    GRU is a simplified version of LSTM with 3 gates instead of 4:
    - Reset gate: r_t = σ(W_r·[h_{t-1}, x_t] + b_r)
    - Update gate: z_t = σ(W_z·[h_{t-1}, x_t] + b_z)
    - Candidate: h̃_t = tanh(W·[r_t⊙h_{t-1}, x_t] + b)
    - Hidden: h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
    
    Key differences from LSTM:
    - No separate cell state (only hidden state)
    - 3 weight matrices instead of 4
    - Update gate controls both forget and input
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize GRU weights.
        
        TODO 1: Initialize 3 weight matrices (reset, update, candidate)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # TODO 2: Reset gate weights
        # self.Wr = ...
        # self.br = ...
        
        # TODO 3: Update gate weights
        # self.Wz = ...
        # self.bz = ...
        
        # TODO 4: Candidate weights
        # self.Wh = ...
        # self.bh = ...
        
        # TODO 5: Output weights
        # self.Wy = ...
        # self.by = ...
        
        pass
    
    def sigmoid(self, x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, targets, h_prev):
        """
        Forward pass through GRU.
        
        TODO 6: Implement GRU forward pass
        
        For each time step:
        1. Compute reset gate: r_t = σ(W_r·[h_{t-1}, x_t] + b_r)
        2. Compute update gate: z_t = σ(W_z·[h_{t-1}, x_t] + b_z)
        3. Compute candidate: h̃_t = tanh(W·[r_t⊙h_{t-1}, x_t] + b)
        4. Update hidden: h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
        5. Compute output and loss
        
        Note: No cell state in GRU! Only hidden state.
        """
        # TODO: Implement forward pass
        pass
    
    def backward(self):
        """
        Backward pass (BPTT) through GRU.
        
        TODO 7: Implement GRU backward pass
        
        Hints:
        - Gradient flows only through hidden state (no cell state)
        - Need to backprop through reset and update gates
        - Don't forget gradient clipping!
        """
        # TODO: Implement backward pass
        pass
    
    def update_weights(self, learning_rate):
        """
        Update weights using computed gradients.
        
        TODO 8: Update all weight matrices
        """
        # TODO: Implement weight update
        pass


def count_parameters(model):
    """
    Count total number of parameters in model.
    
    TODO 9: Count parameters in each weight matrix
    """
    total = 0
    
    if isinstance(model, LSTM):
        # TODO: Count LSTM parameters (Wf, Wi, Wc, Wo, Wy + biases)
        pass
    elif isinstance(model, GRU):
        # TODO: Count GRU parameters (Wr, Wz, Wh, Wy + biases)
        pass
    
    return total


def train_model(model, text, char_to_idx, num_iterations=1000):
    """
    Train a model and return timing + loss info.
    
    TODO 10: Implement training loop
    - Track loss
    - Track time per iteration
    - Return both
    """
    losses = []
    times = []
    
    # TODO: Implement training
    
    return losses, times


def compare_models(text, char_to_idx, hidden_size=64):
    """
    Compare LSTM and GRU side by side.
    
    TODO 11: Train both models and collect metrics
    """
    vocab_size = len(char_to_idx)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # TODO 12: Create LSTM
    print("\n1. Training LSTM...")
    # lstm = LSTM(vocab_size, hidden_size, vocab_size)
    # lstm_params = count_parameters(lstm)
    # lstm_losses, lstm_times = train_model(lstm, text, char_to_idx)
    
    # TODO 13: Create GRU
    print("\n2. Training GRU...")
    # gru = GRU(vocab_size, hidden_size, vocab_size)
    # gru_params = count_parameters(gru)
    # gru_losses, gru_times = train_model(gru, text, char_to_idx)
    
    results = {
        'lstm': {
            'params': 0,  # lstm_params,
            'losses': [],  # lstm_losses,
            'times': []  # lstm_times
        },
        'gru': {
            'params': 0,  # gru_params,
            'losses': [],  # gru_losses,
            'times': []  # gru_times
        }
    }
    
    return results


def plot_comparison(results):
    """
    Create comparison plots.
    
    TODO 14: Plot:
    - Training loss (both models)
    - Training speed (time per iteration)
    - Parameter count (bar chart)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # TODO: Plot 1 - Loss curves
    
    # TODO: Plot 2 - Speed comparison
    
    # TODO: Plot 3 - Parameter count
    
    plt.tight_layout()
    plt.show()


def write_comparison_report(results):
    """
    Write detailed comparison report.
    
    TODO 15: Answer these questions:
    
    1. Parameters:
       - How many parameters does each model have?
       - What's the percentage difference?
    
    2. Training Speed:
       - Which model trains faster?
       - By how much (percentage)?
    
    3. Final Performance:
       - Which model achieves lower final loss?
       - Is the difference significant?
    
    4. Convergence:
       - Which model converges faster?
       - How many iterations to reach threshold?
    
    5. Recommendations:
       - When would you use LSTM?
       - When would you use GRU?
       - What are the trade-offs?
    """
    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    
    print("\nYour report here...")


if __name__ == "__main__":
    print(__doc__)
    
    # Load data
    with open('../data/input.txt', 'r') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    print(f"Training on {len(text)} characters")
    print(f"Vocabulary size: {len(chars)}")
    
    # TODO 16: Run comparison
    # results = compare_models(text, char_to_idx)
    
    # TODO 17: Plot results
    # plot_comparison(results)
    
    # TODO 18: Write report
    # write_comparison_report(results)
    
    print("\n✅ Exercise 5 complete!")
    print("Compare with solutions/solution_05_lstm_vs_gru.py")
    
    print("\n" + "=" * 60)
    print("CONGRATULATIONS!")
    print("=" * 60)
    print("\nYou've completed all 5 exercises!")
    print("You now have a deep understanding of:")
    print("  - LSTM architecture")
    print("  - Gate behavior")
    print("  - Component importance")
    print("  - Memory capacity")
    print("  - LSTM vs GRU trade-offs")
    print("\nYou're ready for Day 3! 🚀")
