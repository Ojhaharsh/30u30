"""
Exercise 4: Monte Carlo Predictions (Ensemble Wisdom)
======================================================

Goal: Run the same network multiple times and aggregate predictions.

Your Task:
- Sample weights from the posterior multiple times
- Run forward passes with different weight samples
- Aggregate predictions into mean and uncertainty
- Compare with single-pass predictions

Learning Objectives:
1. Monte Carlo approximation of Bayesian inference
2. Why multiple samples = better uncertainty
3. Law of large numbers: more samples → better estimates
4. Computational cost vs accuracy tradeoff

Time: 20-30 minutes
Difficulty: Medium
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def generate_sine_data(n_train=150):
    """
    Generate sine wave data for demonstration.
    
    Args:
        n_train: Number of training points
        
    Returns:
        X_train: Input features (Nx1)
        y_train: Target values
        X_test: Test inputs (densely sampled)
        y_test: Noiseless test targets
    """
    # TODO 1: Sample training points randomly
    # X_train = np.random.uniform(0, 4*np.pi, n_train)
    # y_train = np.sin(X_train) + np.random.randn(n_train) * 0.1
    
    # TODO 2: Generate dense test points
    # X_test = np.linspace(0, 4*np.pi, 500)
    # y_test = np.sin(X_test)
    
    # TODO 3: Reshape to column vectors
    # X_train = X_train[:, np.newaxis]
    # X_test = X_test[:, np.newaxis]
    
    # TODO 4: Sort training data by X for plotting
    # idx = np.argsort(X_train.flatten())
    # X_train = X_train[idx]
    # y_train = y_train[idx]
    
    # TODO 5: Return data
    # return X_train, y_train, X_test, y_test
    
    pass


def monte_carlo_forward(model, X, n_samples=100):
    """
    Run forward pass multiple times with different weight samples.
    
    Args:
        model: Trained BayesianLinearNetwork
        X: Input data (Nx1)
        n_samples: Number of MC samples (forward passes)
        
    Returns:
        predictions: (N, n_samples) array of predictions
        mean: Mean prediction at each point
        std: Standard deviation of predictions
        
    Why this works:
    - Each time we call forward(), the network samples new weights
    - Different weight samples → different predictions
    - The collection of predictions approximates the posterior
    - Mean = best estimate, Std = uncertainty
    """
    
    # TODO 6: Initialize array to store predictions
    # predictions = np.zeros((len(X), n_samples))
    
    # TODO 7: Loop n_samples times
    # for i in range(n_samples):
    
    # TODO 8: Run forward pass once (new weight samples each time)
    # predictions[:, i] = model.forward(X).flatten()
    
    # TODO 9: Compute mean of predictions (across samples, not features)
    # mean = np.mean(predictions, axis=1)
    
    # TODO 10: Compute std of predictions
    # std = np.std(predictions, axis=1)
    
    # TODO 11: Return results
    # return predictions, mean, std
    
    pass


def analyze_mc_convergence():
    """
    Study how many samples are needed for good uncertainty estimates.
    """
    print("Training initial model...")
    
    # TODO 12: Create and train a Bayesian network
    # X_train, y_train, X_test, y_test = generate_sine_data(n_train=100)
    # model = BayesianLinearNetwork(
    #     input_dim=1,
    #     hidden_dims=[32, 16],
    #     output_dim=1,
    #     kl_weight=0.01
    # )
    # losses = model.train(X_train, y_train, epochs=400, lr=0.01, verbose=False)
    
    print("Running MC predictions with different sample sizes...")
    
    # TODO 13: Test different numbers of MC samples
    # sample_sizes = [1, 5, 10, 20, 50, 100, 200]
    
    results = {
        'n_samples': [],
        'std_estimates': [],
        'std_stability': []  # How much std varies with different runs
    }
    
    # TODO 14: For each sample size:
    # for n_mc in sample_sizes:
    #     print(f"  n_samples = {n_mc:3d}", end="", flush=True)
    
    # TODO 15: Run MC forward pass
    # _, mean, std = monte_carlo_forward(model, X_test, n_samples=n_mc)
    
    # TODO 16: Run multiple times to check stability
    # Run this again 3 times and see how much std varies
    # stds_multiple_runs = []
    # for _ in range(3):
    #     _, _, std_run = monte_carlo_forward(model, X_test, n_samples=n_mc)
    #     stds_multiple_runs.append(std_run)
    
    # TODO 17: Compute variation in std estimates
    # std_variation = np.std(stds_multiple_runs, axis=0)
    # mean_variation = np.mean(std_variation)
    
    # TODO 18: Store results
    # results['n_samples'].append(n_mc)
    # results['std_estimates'].append(np.mean(std))
    # results['std_stability'].append(mean_variation)
    
    # TODO 19: Print interpretation
    # print(f" → Uncertainty={np.mean(std):.4f}, Stability={mean_variation:.4f}")
    
    # Visualization
    print("\n\nPlotting MC convergence...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: How uncertainty estimate converges
    ax = axes[0]
    if 'results' in locals() and results['n_samples']:
        ax.semilogx(results['n_samples'], results['std_estimates'], 'o-', 
                    linewidth=2, markersize=8)
        ax.set_xlabel('Number of MC Samples')
        ax.set_ylabel('Mean Predicted Uncertainty')
        ax.set_title('Convergence of Uncertainty Estimates')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Stability of estimates
    ax = axes[1]
    if 'results' in locals() and results['n_samples']:
        ax.loglog(results['n_samples'], results['std_stability'], 's-', 
                  linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Number of MC Samples')
        ax.set_ylabel('Variation in Uncertainty')
        ax.set_title('Stability: Lower = More Reliable')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'mc_convergence.png', dpi=150, bbox_inches='tight')
    print("  Saved to: mc_convergence.png")
    plt.close()
    
    print("\n[ok] MC convergence analysis completed.")
    print("\nKey insights:")
    print("  - 1 sample: Just a point estimate (not Bayesian!)")
    print("  - 5-10 samples: Basic uncertainty estimates")
    print("  - 50+ samples: Good convergence (law of large numbers)")
    print("  - 100+ samples: Estimates stabilize nicely")
    print("\nRule of thumb: Use 50-100 MC samples for inference")


if __name__ == "__main__":
    analyze_mc_convergence()
