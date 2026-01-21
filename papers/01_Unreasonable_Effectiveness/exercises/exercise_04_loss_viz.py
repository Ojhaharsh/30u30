"""
Exercise 4: Loss Visualization
===============================

Goal: Visualize training dynamics and diagnose problems.

Your Task:
- Log losses during training
- Create visualizations
- Identify training patterns
- Diagnose and fix issues

Learning Objectives:
1. How to monitor training progress
2. Recognize common problems (explosions, plateaus)
3. When to stop training
4. Debugging techniques

Time: 15-30 minutes
Difficulty: Quick ⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import CharRNN


def train_with_logging(rnn, data, char_to_idx, num_iterations=1000):
    """
    Train RNN and log losses at each iteration.
    
    Args:
        rnn: CharRNN model
        data: Training text
        char_to_idx: Character to index mapping
        num_iterations: Number of training iterations
    
    Returns:
        losses: List of loss values
    
    TODO 1: Implement training loop with loss logging
    """
    losses = []
    seq_length = 25
    learning_rate = 0.01
    
    h_prev = np.zeros(rnn.hidden_size)
    
    print("Training with loss logging...")
    
    for iteration in range(num_iterations):
        # TODO: Sample sequence
        # TODO: Forward pass
        # TODO: Backward pass
        # TODO: Update weights
        # TODO: Log loss
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}")
    
    return losses


def plot_loss_curve(losses, save_path=None):
    """
    Plot the training loss curve.
    
    TODO 2: Create a clear loss visualization
    """
    plt.figure(figsize=(12, 6))
    
    # TODO: Plot raw losses
    # plt.plot(losses, alpha=0.3, label='Raw loss')
    
    # TODO: Plot smoothed losses (moving average)
    # window = 50
    # smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    # plt.plot(range(window-1, len(losses)), smoothed, linewidth=2, label='Smoothed')
    
    # TODO: Add labels, legend, grid
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def identify_training_patterns(losses):
    """
    Analyze loss curve and identify patterns.
    
    TODO 3: Identify these patterns in your loss curve:
    
    1. Smooth decrease - Good training
    2. Plateau - Learning stopped
    3. Spikes - Gradient explosion
    4. Oscillation - Unstable training
    5. Sudden drop - Breakthrough moment
    """
    print("\n" + "=" * 60)
    print("TRAINING PATTERN ANALYSIS")
    print("=" * 60)
    
    # TODO: Compute statistics
    # initial_loss = losses[0]
    # final_loss = losses[-1]
    # min_loss = min(losses)
    # max_loss = max(losses)
    
    print("\nStatistics:")
    print("  TODO: Add your statistics")
    
    print("\nObserved patterns:")
    print("  TODO: Describe what you see")


def diagnose_problems(losses):
    """
    Diagnose common training problems.
    
    TODO 4: Check for these issues:
    
    1. Gradient explosion
       - Symptoms: Sudden spikes in loss
       - Fix: Reduce learning rate or add gradient clipping
    
    2. Learning rate too high
       - Symptoms: Loss oscillates
       - Fix: Reduce learning rate by 10x
    
    3. Learning rate too low
       - Symptoms: Very slow decrease
       - Fix: Increase learning rate by 2-5x
    
    4. Model capacity too small
       - Symptoms: Quick plateau, high final loss
       - Fix: Increase hidden_size
    
    5. Overfitting
       - Symptoms: Training loss ↓ but validation loss ↑
       - Fix: Add regularization or early stopping
    """
    print("\n" + "=" * 60)
    print("PROBLEM DIAGNOSIS")
    print("=" * 60)
    
    # TODO: Check for gradient explosion
    print("\n1. Gradient explosion:")
    # spikes = count_spikes(losses)
    # if spikes > threshold:
    #     print("   ⚠️ DETECTED!")
    # else:
    #     print("   ✅ OK")
    
    # TODO: Check for other issues
    print("   TODO: Add your diagnosis")


def suggest_fixes(losses):
    """
    Suggest improvements based on loss pattern.
    
    TODO 5: Provide specific recommendations
    """
    print("\n" + "=" * 60)
    print("SUGGESTED IMPROVEMENTS")
    print("=" * 60)
    
    print("\nBased on your loss curve:")
    print("  TODO: Add specific suggestions")
    
    print("\nExample fixes:")
    print("  - For explosions: Add gradient clipping")
    print("  - For oscillations: Reduce learning rate")
    print("  - For plateaus: Increase model size")


def compare_training_runs():
    """
    Compare multiple training runs.
    
    TODO 6 (BONUS): Train with different settings and compare
    """
    print("\n" + "=" * 60)
    print("COMPARING TRAINING RUNS")
    print("=" * 60)
    
    # TODO: Train with different hyperparameters
    # runs = {
    #     'baseline': train_with_logging(...),
    #     'high_lr': train_with_logging(..., lr=0.1),
    #     'low_lr': train_with_logging(..., lr=0.001),
    # }
    
    # TODO: Plot all on same graph
    # for name, losses in runs.items():
    #     plt.plot(losses, label=name)
    
    pass


def create_diagnostic_dashboard(losses):
    """
    Create comprehensive diagnostic visualization.
    
    TODO 7 (BONUS): Create multi-panel dashboard
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Loss curve
    # axes[0, 0].plot(losses)
    # axes[0, 0].set_title('Loss Curve')
    
    # Panel 2: Loss histogram
    # axes[0, 1].hist(losses, bins=50)
    # axes[0, 1].set_title('Loss Distribution')
    
    # Panel 3: Rolling statistics
    # rolling_mean = ...
    # rolling_std = ...
    # axes[1, 0].plot(rolling_mean, label='Mean')
    # axes[1, 0].fill_between(..., label='±1 std')
    
    # Panel 4: Gradient magnitude (if available)
    # axes[1, 1].plot(gradient_norms)
    # axes[1, 1].set_title('Gradient Magnitude')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    
    # TODO 8: Load data and create model
    # data, char_to_idx, idx_to_char = load_data('data/input.txt')
    # vocab_size = len(char_to_idx)
    # rnn = CharRNN(vocab_size, hidden_size=128)
    
    # Train with logging
    # losses = train_with_logging(rnn, data, char_to_idx)
    
    # Plot loss curve
    # plot_loss_curve(losses, save_path='loss_curve.png')
    
    # Analyze patterns
    # identify_training_patterns(losses)
    
    # Diagnose problems
    # diagnose_problems(losses)
    
    # Suggest fixes
    # suggest_fixes(losses)
    
    # Bonus: Compare runs
    # compare_training_runs()
    
    # Bonus: Diagnostic dashboard
    # create_diagnostic_dashboard(losses)
    
    print("\n✅ Exercise 4 complete!")
    print("Compare with solutions/exercise_04_solution.py")
