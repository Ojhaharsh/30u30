"""
Exercise 3: The "Beta Parameter" Study (Calibrating Uncertainty)
=================================================================

Goal: Explore how KL weight controls the uncertainty calibration.

Your Task:
- Train networks with different kl_weight values
- Measure how uncertainty changes
- Find the "sweet spot" for calibration
- Plot the uncertainty-accuracy tradeoff

Learning Objectives:
1. What does kl_weight do? (controls prior strength)
2. Low kl_weight → Model is confident but overfit
3. High kl_weight → Model is uncertain but underfits
4. How to find the optimal balance

Time: 30-40 minutes
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def create_sine_dataset(n_points=200, noise_std=0.1):
    """
    Create a noisy sine wave dataset.
    
    Args:
        n_points: Number of training points
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        X_train: Training inputs
        y_train: Training targets (noisy sine)
        X_test: Test inputs
        y_test: Test targets (noiseless sine)
    """
    # TODO 1: Generate training points uniformly across [0, 4π]
    # X_train = np.random.uniform(0, 4*np.pi, n_points)
    
    # TODO 2: Generate noisy sine wave
    # y_train = np.sin(X_train) + np.random.randn(n_points) * noise_std
    
    # TODO 3: Generate test points (denser, noiseless)
    # X_test = np.linspace(0, 4*np.pi, 500)
    # y_test = np.sin(X_test)
    
    # TODO 4: Sort training data by X for plotting
    # idx = np.argsort(X_train)
    # X_train = X_train[idx]
    # y_train = y_train[idx]
    
    # TODO 5: Return as column vectors (reshape to Nx1)
    # return X_train[:, np.newaxis], y_train, X_test[:, np.newaxis], y_test
    
    pass


def evaluate_model(model, X_test, y_test, n_mc=50):
    """
    Evaluate a trained model and compute metrics.
    
    Args:
        model: Trained BayesianLinearNetwork
        X_test: Test inputs
        y_test: Test targets (noiseless)
        n_mc: Number of MC samples for uncertainty
        
    Returns:
        mse: Mean squared error
        mean_uncertainty: Average predictive uncertainty
        calibration_error: How well-calibrated are the uncertainties?
    """
    # TODO 6: Run network multiple times
    # predictions = np.zeros((len(X_test), n_mc))
    # for i in range(n_mc):
    #     predictions[:, i] = model.forward(X_test).flatten()
    
    # TODO 7: Compute mean and std
    # mean_pred = np.mean(predictions, axis=1)
    # std_pred = np.std(predictions, axis=1)
    
    # TODO 8: Compute MSE
    # mse = np.mean((mean_pred - y_test)**2)
    
    # TODO 9: Compute mean uncertainty (average std)
    # mean_uncertainty = np.mean(std_pred)
    
    # TODO 10: Compute calibration error
    # Compute how many points fall within 1σ band (should be ~68%)
    # in_band = np.abs(mean_pred - y_test) <= std_pred
    # coverage_1sigma = np.mean(in_band)
    # calibration_error = np.abs(coverage_1sigma - 0.68)
    
    # TODO 11: Return metrics
    # return mse, mean_uncertainty, calibration_error
    
    pass


def study_kl_weights():
    """
    Train models with different kl_weight values and compare.
    """
    print("Preparing dataset...")
    X_train, y_train, X_test, y_test = create_sine_dataset(n_points=150, noise_std=0.15)
    
    # TODO 12: Define kl_weights to test
    kl_weights = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    # This ranges from "no regularization" to "very strong regularization"
    
    results = {
        'kl_weights': [],
        'mse': [],
        'uncertainty': [],
        'calibration_error': []
    }
    
    print("\nTraining models with different KL weights...\n")
    
    # TODO 13: For each kl_weight:
    for kl_w in kl_weights:
        print(f"  kl_weight = {kl_w:8.4f}", end="", flush=True)
        
        # TODO 14: Create and train model
        # model = BayesianLinearNetwork(
        #     input_dim=1,
        #     hidden_dims=[32, 16],
        #     output_dim=1,
        #     kl_weight=kl_w
        # )
        # losses = model.train(X_train, y_train, epochs=300, lr=0.01, verbose=False)
        
        # TODO 15: Evaluate on test set
        # mse, unc, calib = evaluate_model(model, X_test, y_test)
        
        # TODO 16: Store results
        # results['kl_weights'].append(kl_w)
        # results['mse'].append(mse)
        # results['uncertainty'].append(unc)
        # results['calibration_error'].append(calib)
        
        # TODO 17: Print results with emoji interpretation
        # print(f" → MSE={mse:.4f}, Uncertainty={unc:.4f}")
        
        pass
    
    # Visualization
    print("\n\nPlotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: MSE vs kl_weight
    ax = axes[0]
    if 'results' in locals() and results['kl_weights']:
        ax.semilogx(results['kl_weights'], results['mse'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('KL Weight (β)')
        ax.set_ylabel('Test MSE')
        ax.set_title('Fit Quality: Higher MSE = Underfitting')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty vs kl_weight
    ax = axes[1]
    if 'results' in locals() and results['kl_weights']:
        ax.semilogx(results['kl_weights'], results['uncertainty'], 's-', 
                    linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('KL Weight (β)')
        ax.set_ylabel('Mean Uncertainty (σ)')
        ax.set_title('Uncertainty: Higher = Network Less Confident')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Calibration vs kl_weight
    ax = axes[2]
    if 'results' in locals() and results['kl_weights']:
        ax.semilogx(results['kl_weights'], results['calibration_error'], '^-', 
                    linewidth=2, markersize=8, color='green')
        ax.axhline(0, color='red', linestyle='--', label='Perfect calibration')
        ax.set_xlabel('KL Weight (β)')
        ax.set_ylabel('Calibration Error')
        ax.set_title('Calibration: Lower = Better Uncertainty Estimates')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'beta_study.png', dpi=150, bbox_inches='tight')
    print("  Saved to: beta_study.png")
    plt.close()
    
    print("\n✓ Beta study completed!")
    print("\nKey insights:")
    print("  - Low β: Model overfits (low MSE but bad uncertainty)")
    print("  - High β: Model underfits (high MSE but better calibration)")
    print("  - Sweet spot: Where MSE and calibration error balance")


if __name__ == "__main__":
    study_kl_weights()
