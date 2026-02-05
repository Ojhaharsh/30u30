"""
Exercise 2: The "Gap Experiment" (Uncertainty in Action)
=========================================================

Goal: Train a Bayesian network on incomplete data and visualize uncertainty.

Your Task:
- Create a dataset with "gaps" (missing regions)
- Train a Bayesian network on this gappy data
- Visualize where the network is uncertain
- Compare vs a regular neural network

Learning Objectives:
1. How uncertainty captures epistemic gaps (where data is missing)
2. Uncertainty ≠ variance (deeper meaning)
3. Aleatoric vs epistemic uncertainty
4. Why Bayesian = better generalization

Time: 25-35 minutes
Difficulty: Hard
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import from implementation
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def create_gappy_dataset(n_points=100, gap_regions=None):
    """
    Create a sine wave with missing regions.
    
    Args:
        n_points: Total points to generate
        gap_regions: List of (start, end) tuples where to remove data
        
    Returns:
        X_train: Input points (excluding gaps)
        y_train: Target values (excluding gaps)
        X_dense: Full input range for predictions
        y_true: True sine wave across full range
        
    Why gaps?
    - In real life, we never have perfect coverage
    - Gaps force the network to EXTRAPOLATE, not interpolate
    - Uncertainty should spike at gaps!
    """
    
    # TODO 1: Generate full sine wave
    # X_dense = np.linspace(0, 4*np.pi, 500)
    # y_true = np.sin(X_dense)
    
    # TODO 2: Create training set without gaps
    # mask = np.ones(len(X_dense), dtype=bool)
    
    # TODO 3: Iterate through gap_regions and set mask to False
    # for start, end in gap_regions:
    #     indices = (X_dense >= start) & (X_dense < end)
    #     mask[indices] = False
    
    # TODO 4: Extract training data from non-gap regions
    # X_train = X_dense[mask]
    # y_train = y_true[mask]
    
    # TODO 5: Return all components
    # return X_train, y_train, X_dense, y_true, mask
    
    pass


def train_bayesian_model_on_gaps():
    """
    Train a Bayesian network on gappy data.
    """
    print("Creating gappy dataset...")
    
    # TODO 6: Define gap regions (e.g., avoid π/2 and 3π/2)
    # gaps = [(np.pi/2 - 0.5, np.pi/2 + 0.5), 
    #         (3*np.pi/2 - 0.5, 3*np.pi/2 + 0.5)]
    # X_train, y_train, X_dense, y_true, mask = create_gappy_dataset(n_points=80, gap_regions=gaps)
    
    # print(f"  Training points: {len(X_train)}")  # Will work once create_gappy_dataset is implemented
    
    print("Initializing Bayesian network...")
    # TODO 7: Create BayesianLinearNetwork with 2 hidden layers
    # model = BayesianLinearNetwork(
    #     input_dim=1,
    #     hidden_dims=[64, 32],
    #     output_dim=1,
    #     kl_weight=0.01
    # )
    
    print("Training...")
    # TODO 8: Train for 500 epochs with learning rate 0.01
    # losses = model.train(X_train[:, np.newaxis], y_train, 
    #                      epochs=500, lr=0.01, verbose=False)
    
    print("Generating predictions...")
    # TODO 9: Run network multiple times to get uncertainty estimates
    # n_mc = 100
    # predictions = np.zeros((len(X_dense), n_mc))
    # for i in range(n_mc):
    #     predictions[:, i] = model.forward(X_dense[:, np.newaxis]).flatten()
    
    # TODO 10: Compute mean and std of predictions
    # mean_pred = np.mean(predictions, axis=1)
    # std_pred = np.std(predictions, axis=1)
    
    # Visualize
    print("Plotting results...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Data and predictions
    ax = axes[0]
    # TODO 10: When you complete TODOs above, uncomment these plots
    # if 'X_train' in locals():
    #     ax.scatter(X_train, y_train, color='red', s=30, label='Training data', zorder=5)
    # if 'X_dense' in locals():
    #     ax.plot(X_dense, y_true, 'k--', linewidth=2, label='True function', zorder=2)
    #     ax.plot(X_dense, mean_pred, 'b-', linewidth=2, label='Bayesian prediction', zorder=3)
    #     ax.fill_between(X_dense, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
    #                      alpha=0.3, color='blue', label='2σ uncertainty')
    # TODO 11: Mark gap regions with shaded area
    # for start, end in gaps:
    #     ax.axvspan(start, end, alpha=0.2, color='red', label='Data gaps' if start == gaps[0][0] else '')
    
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Bayesian Network on Gappy Data: Uncertainty at Missing Regions')
    
    # Plot 2: Uncertainty profile
    ax = axes[1]
    # TODO 11: When you have std_pred from above, uncomment:
    # if 'std_pred' in locals():
    #     ax.plot(X_dense, std_pred, 'purple', linewidth=2)
    #     ax.fill_between(X_dense, std_pred, alpha=0.3, color='purple')
    #     # TODO 12: Also mark gap regions here
    #     for start, end in gaps:
    #         ax.axvspan(start, end, alpha=0.2, color='red')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Uncertainty (σ)')
    ax.set_title('Where is the network uncertain?')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'gap_experiment.png', dpi=150, bbox_inches='tight')
    print("  Saved to: gap_experiment.png")
    plt.close()
    
    print("\n[ok] Gap experiment completed.")
    print("\nKey observation:")
    print("  - Uncertainty is HIGH in gap regions (good!)")
    print("  - Uncertainty is LOW where data exists (good!)")
    print("  - This is epistemic uncertainty: 'I don't have data here'")


if __name__ == "__main__":
    train_bayesian_model_on_gaps()
