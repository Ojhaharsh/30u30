"""
Exercise 5: Advanced MDL (Compression Ratio Analysis)
======================================================

Goal: Analyze the full MDL objective and understand the compression frontier.

Your Task:
- Compute data fit loss and KL divergence separately
- Calculate effective compression ratio
- Plot the two terms against each other
- Find the Pareto frontier of compression

Learning Objectives:
1. MDL = -log(data fit) - log(prior knowledge)
2. Tradeoff: how much accuracy to trade for compression?
3. Pareto efficiency: can't improve both simultaneously
4. Real-world applications (edge devices, mobile)

Time: 30-40 minutes
Difficulty: Hard ⏱️⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def analyze_loss_components(model, X_test, y_test):
    """
    Decompose the loss into data fit and regularization terms.
    
    Args:
        model: Trained BayesianLinearNetwork
        X_test: Test inputs
        y_test: Test targets
        
    Returns:
        reconstruction_loss: How well does it fit the data?
        kl_divergence: How far is it from the prior?
        mdl_loss: Combined loss (lower = better compression)
        
    The MDL principle:
        L_MDL = L_data + KL_divergence
        
    This balances two goals:
        1. L_data: Explain the data well (small residuals)
        2. KL: Stay close to prior (simple weights)
    
    A good model achieves BOTH!
    """
    
    # TODO 1: Forward pass
    # predictions = model.forward(X_test)
    
    # TODO 2: Compute reconstruction loss (MSE)
    # reconstruction_loss = np.mean((predictions - y_test)**2)
    
    # TODO 3: Compute KL divergence from model parameters
    # This requires accessing the network's weight distributions
    # kl_divergence = model.compute_kl_divergence()
    
    # TODO 4: Combine into MDL loss
    # mdl_loss = reconstruction_loss + kl_divergence
    
    # TODO 5: Return components
    # return reconstruction_loss, kl_divergence, mdl_loss
    
    pass


def study_pareto_frontier():
    """
    Train networks with different kl_weights and plot the tradeoff.
    """
    print("Generating dataset...")
    # TODO 6: Create a dataset (sine wave with noise)
    X_train = np.random.uniform(0, 4*np.pi, 200)[:, np.newaxis]
    y_train = np.sin(X_train.flatten()) + np.random.randn(200) * 0.15
    
    X_test = np.linspace(0, 4*np.pi, 500)[:, np.newaxis]
    y_test = np.sin(X_test.flatten())
    
    # TODO 7: Define kl_weights spanning the compression frontier
    kl_weights = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    pareto_data = {
        'kl_weights': [],
        'reconstruction_loss': [],
        'kl_divergence': [],
        'mdl_loss': [],
        'compression_ratio': []
    }
    
    print("\nTraining models across compression frontier...\n")
    
    # TODO 8: For each kl_weight:
    for kl_w in kl_weights:
        print(f"  β = {kl_w:8.5f}", end="", flush=True)
        
        # TODO 9: Train model
        # model = BayesianLinearNetwork(
        #     input_dim=1,
        #     hidden_dims=[64, 32],
        #     output_dim=1,
        #     kl_weight=kl_w
        # )
        # losses = model.train(X_train, y_train, epochs=500, lr=0.01, verbose=False)
        
        # TODO 10: Analyze loss components
        # recon_loss, kl_div, mdl_loss = analyze_loss_components(model, X_test, y_test)
        
        # TODO 11: Compute compression ratio (lower = better compression)
        # compression_ratio = kl_div / (recon_loss + 1e-8)
        
        # TODO 12: Store results
        # pareto_data['kl_weights'].append(kl_w)
        # pareto_data['reconstruction_loss'].append(recon_loss)
        # pareto_data['kl_divergence'].append(kl_div)
        # pareto_data['mdl_loss'].append(mdl_loss)
        # pareto_data['compression_ratio'].append(compression_ratio)
        
        # TODO 13: Print with interpretation
        # print(f" → Recon={recon_loss:.4f}, KL={kl_div:.4f}, Ratio={compression_ratio:.2f}")
        
        pass
    
    # Visualization
    print("\n\nPlotting Pareto frontier...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: The Pareto frontier (reconstruction vs KL)
    ax = axes[0, 0]
    if 'pareto_data' in locals() and pareto_data['reconstruction_loss']:
        ax.scatter(pareto_data['reconstruction_loss'], 
                   pareto_data['kl_divergence'], 
                   s=100, c=range(len(pareto_data['kl_weights'])), 
                   cmap='viridis', alpha=0.7)
        for i, (x, y, kl_w) in enumerate(zip(pareto_data['reconstruction_loss'],
                                             pareto_data['kl_divergence'],
                                             pareto_data['kl_weights'])):
            if i % 2 == 0:  # Label every other point
                ax.annotate(f'β={kl_w:.2e}', (x, y), fontsize=8)
        ax.set_xlabel('Reconstruction Loss (fit to data)')
        ax.set_ylabel('KL Divergence (regularization)')
        ax.set_title('Pareto Frontier: Compression Tradeoff')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: MDL loss across weights
    ax = axes[0, 1]
    if 'pareto_data' in locals() and pareto_data['mdl_loss']:
        ax.semilogx(pareto_data['kl_weights'], pareto_data['mdl_loss'], 'o-', linewidth=2)
        ax.set_xlabel('KL Weight (β)')
        ax.set_ylabel('MDL Loss')
        ax.set_title('Total Compression Cost (lower = better)')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Compression ratio
    ax = axes[1, 0]
    if 'pareto_data' in locals() and pareto_data['compression_ratio']:
        ax.loglog(pareto_data['kl_weights'], pareto_data['compression_ratio'], 's-', 
                  linewidth=2, color='purple')
        ax.set_xlabel('KL Weight (β)')
        ax.set_ylabel('Compression Ratio (KL/Recon)')
        ax.set_title('Model Simplicity vs Data Fit')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Component breakdown
    ax = axes[1, 1]
    if 'pareto_data' in locals() and pareto_data['kl_weights']:
        width = 0.35
        x = np.arange(len(pareto_data['kl_weights']))
        ax.bar(x - width/2, pareto_data['reconstruction_loss'], width, label='Data Fit')
        ax.bar(x + width/2, pareto_data['kl_divergence'], width, label='Regularization')
        ax.set_xlabel('KL Weight (β)')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Component Breakdown')
        # TODO 14: Set xticks to show kl_weights
        # ax.set_xticks(x)
        # ax.set_xticklabels([f'{kl_w:.2e}' for kl_w in pareto_data['kl_weights']], 
        #                    rotation=45, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
    print("  Saved to: pareto_frontier.png")
    plt.close()
    
    print("\n✓ Pareto frontier analysis completed!")
    print("\nKey insights:")
    print("  - LEFT side: Underfitting (KL too high)")
    print("  - RIGHT side: Overfitting (Recon too high)")
    print("  - MIDDLE: Sweet spot (balanced compression)")
    print("\nPareto optimal = can't improve both objectives simultaneously!")


if __name__ == "__main__":
    study_pareto_frontier()
