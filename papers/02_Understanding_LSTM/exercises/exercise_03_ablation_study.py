"""
Exercise 3: Ablation Study
===========================

Goal: Remove each gate one at a time and measure the impact.

Your Task:
- Implement 4 LSTM variants (no forget, no input, no output, baseline)
- Train all 4 on the same task
- Compare performance metrics
- Explain which gates are most critical

Learning Objectives:
1. Understand which gates are essential
2. See how gates complement each other
3. Learn trade-offs between complexity and performance
4. Gain intuition for LSTM design choices

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM


class LSTMNoForget(LSTM):
    """LSTM with forget gate fixed to 1 (always remember)."""
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        Modified forward pass with f_t = 1.
        
        TODO 1: Copy the forward method from LSTM but set f_t = np.ones(...)
        """
        # Your code here
        pass


class LSTMNoInput(LSTM):
    """LSTM with input gate fixed to 1 (always add)."""
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        Modified forward pass with i_t = 1.
        
        TODO 2: Copy the forward method from LSTM but set i_t = np.ones(...)
        """
        # Your code here
        pass


class LSTMNoOutput(LSTM):
    """LSTM with output gate fixed to 1 (always output everything)."""
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        Modified forward pass with o_t = 1.
        
        TODO 3: Copy the forward method from LSTM but set o_t = np.ones(...)
        """
        # Your code here
        pass


def train_model(model, text, char_to_idx, num_iterations=1000):
    """
    Train a model and return loss history.
    
    Args:
        model: LSTM model to train
        text: Training text
        char_to_idx: Character to index mapping
        num_iterations: Number of training iterations
    
    Returns:
        losses: List of loss values
    """
    hidden_size = model.hidden_size
    seq_length = 25
    learning_rate = 0.001
    
    h_prev = np.zeros(hidden_size)
    C_prev = np.zeros(hidden_size)
    
    losses = []
    
    print("Training...")
    for iteration in range(num_iterations):
        # TODO 4: Implement training loop
        # - Sample random sequence
        # - Forward pass
        # - Backward pass
        # - Update weights
        # - Store loss
        
        pass
    
    return losses


def compare_models(text, char_to_idx, hidden_size=64):
    """
    Compare all 4 model variants.
    
    TODO 5: Create and train all 4 models:
    - Baseline LSTM
    - No Forget Gate
    - No Input Gate
    - No Output Gate
    
    Returns:
        results: Dictionary mapping model name to loss history
    """
    vocab_size = len(char_to_idx)
    results = {}
    
    # TODO 6: Create baseline LSTM
    print("\n1. Training baseline LSTM...")
    # baseline = LSTM(...)
    # results['Baseline'] = train_model(...)
    
    # TODO 7: Create and train no-forget LSTM
    print("\n2. Training LSTM without forget gate...")
    # no_forget = LSTMNoForget(...)
    # results['No Forget'] = train_model(...)
    
    # TODO 8: Create and train no-input LSTM
    print("\n3. Training LSTM without input gate...")
    # no_input = LSTMNoInput(...)
    # results['No Input'] = train_model(...)
    
    # TODO 9: Create and train no-output LSTM
    print("\n4. Training LSTM without output gate...")
    # no_output = LSTMNoOutput(...)
    # results['No Output'] = train_model(...)
    
    return results


def plot_comparison(results):
    """
    Plot loss curves for all models.
    
    TODO 10: Create a plot comparing all 4 models
    - Use different colors for each
    - Add legend
    - Label axes
    - Add title
    """
    plt.figure(figsize=(12, 6))
    
    # Your plotting code here
    
    plt.show()


def analyze_results(results):
    """
    Analyze and compare final performance.
    
    TODO 11: Compute and print:
    - Final loss for each model
    - Convergence speed (iterations to reach threshold)
    - Ranking from best to worst
    """
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    # Your analysis here


def write_conclusions():
    """
    Write conclusions about which gates matter most.
    
    TODO 12: Answer these questions:
    1. Which gate removal hurt performance the most?
    2. Why do you think that gate is most important?
    3. Can the LSTM function without any single gate?
    4. What does this tell you about LSTM design?
    """
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    print("\nYour conclusions here...")


if __name__ == "__main__":
    print(__doc__)
    
    # Create training data
    with open('../data/input.txt', 'r') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    print(f"Training on {len(text)} characters")
    print(f"Vocabulary size: {len(chars)}")
    
    # Compare models
    results = compare_models(text, char_to_idx)
    
    # Plot
    plot_comparison(results)
    
    # Analyze
    analyze_results(results)
    
    # Conclusions
    write_conclusions()
    
    print("\n✅ Exercise 3 complete!")
    print("Compare with solutions/solution_03_ablation_study.py")
