"""
Minimal MDL / Bayesian Network Training Script
==============================================

Train a "Noisy Weight" Neural Network to solve a regression task.
We use a "Gappy Sine Wave" dataset to demonstrate the power of MDL.

The Challenge:
-------------
We give the model data from x=[-3, -1] and x=[1, 3].
We delete the data in the middle (x=[-1, 1]).

- A Standard NN will confidently draw a straight line through the gap.
- An MDL Network will show HIGH UNCERTAINTY in the gap.

Usage:
    # Train with default settings
    python train_minimal.py
    
    # Train with strong "Simplicity Pressure" (High KL weight)
    python train_minimal.py --kl-weight 0.5
    
    # Train a massive network (to see if it overfits)
    python train_minimal.py --hidden-size 100 --epochs 5000
"""

import numpy as np
import argparse
import os
import sys

# Ensure we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from implementation import MDLNetwork
from visualization import create_comprehensive_report

def generate_gappy_data(n_samples=100, noise_std=0.1):
    """
    Generate a sine wave with a missing chunk in the middle.
    
    Args:
        n_samples: Number of data points
        noise_std: Amount of random noise to add to y
        
    Returns:
        X: Inputs (n, 1)
        y: Targets (n, 1)
    """
    # Generate random points in [-3, 3]
    X = np.random.uniform(-3, 3, size=(n_samples, 1))
    
    # Create the gap: Remove points where |x| < 1
    # This forces the model to extrapolate in the middle
    mask = np.abs(X) > 1.0
    X = X[mask].reshape(-1, 1)
    
    # Calculate y = sin(x) + noise
    y = np.sin(X) + np.random.normal(0, noise_std, size=X.shape)
    
    print(f"Data Generation:")
    print(f"  - Range: [-3, 3] with GAP in [-1, 1]")
    print(f"  - Samples: {len(X)}")
    print(f"  - Noise Level: {noise_std}")
    
    return X, y

def train(args):
    """
    Main training loop.
    """
    # 1. Prepare Data
    print(f"\n{'='*60}")
    print("1. Generating Synthetic Data")
    print(f"{'='*60}")
    X_train, y_train = generate_gappy_data(n_samples=args.samples)
    
    # 2. Initialize Network
    print(f"\n{'='*60}")
    print("2. Initializing MDL Network")
    print(f"{'='*60}")
    
    net = MDLNetwork(
        input_size=1,
        hidden_size=args.hidden_size,
        output_size=1
    )
    
    print(f"  - Architecture: [1] -> Bayesian[{args.hidden_size}] -> ReLU -> Bayesian[1]")
    print(f"  - Complexity Cost Weight (Beta): {args.kl_weight}")
    print(f"  - Learning Rate: {args.lr}")
    
    # 3. Training Loop
    print(f"\n{'='*60}")
    print("3. Starting Optimization")
    print(f"{'='*60}")
    
    history = {'total': [], 'nll': [], 'kl': []}
    
    # Pre-calculate batch size normalization factor
    # We want KL to be comparable to NLL (which is averaged over batch)
    # So we divide KL by the number of training batches (or total samples)
    # Hint: In the paper, this scaling is crucial!
    complexity_scale = 1.0 / len(X_train)
    
    for epoch in range(args.epochs):
        # --- Forward Pass ---
        # Note: Every time we call forward(), new noise is sampled!
        preds = net.forward(X_train)
        
        # --- Calculate Loss ---
        # 1. Error Cost (Negative Log Likelihood)
        # Assuming Gaussian noise, NLL is proportional to MSE
        mse = np.mean((preds - y_train) ** 2)
        nll = mse 
        
        # 2. Complexity Cost (KL Divergence)
        # How far are our weights from the prior N(0,1)?
        raw_kl = net.total_kl()
        kl_term = raw_kl * complexity_scale
        
        # 3. Total Objective (Free Energy)
        # Minimize: Error + beta * Complexity
        loss = nll + args.kl_weight * kl_term
        
        # --- Backward Pass ---
        # Gradient of MSE w.r.t predictions
        d_nll = 2 * (preds - y_train) / len(X_train)
        
        # Backpropagate
        net.backward(d_nll)
        
        # --- Update Weights ---
        net.update_weights(args.lr, args.kl_weight * complexity_scale)
        
        # --- Logging ---
        history['total'].append(loss)
        history['nll'].append(nll)
        history['kl'].append(kl_term)
        
        if epoch % (args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Total: {loss:.4f} | "
                  f"Error (NLL): {nll:.4f} | "
                  f"Complexity (KL): {kl_term:.4f}")

    print(f"\nTraining complete.")
    print(f"Final Error: {history['nll'][-1]:.4f}")
    
    # 4. Save Model
    net.save(args.checkpoint)
    
    # 5. Generate Analysis Report
    print(f"\n{'='*60}")
    print("4. Generating Visualization Report")
    print(f"{'='*60}")
    
    # Create a dense test set for smooth plotting
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    
    create_comprehensive_report(
        model=net,
        history=history,
        X=X_train,
        y=y_train,
        save_dir=args.output_dir
    )
    
    print(f"\nDone! Check the '{args.output_dir}' folder for your plots.")

def main():
    parser = argparse.ArgumentParser(description='Train an MDL / Bayesian Neural Network')
    
    # Experiment Settings
    parser.add_argument('--samples', type=int, default=150,
                       help='Number of training points (default: 150)')
    parser.add_argument('--hidden-size', type=int, default=20,
                       help='Hidden layer size (default: 20)')
    parser.add_argument('--epochs', type=int, default=3000,
                       help='Training epochs (default: 3000)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    
    # The Most Important Parameter: Beta (KL Weight)
    parser.add_argument('--kl-weight', type=float, default=0.1,
                       help='Simplicity Pressure (Beta). Higher = Simpler/Fuzzier weights. (default: 0.1)')
    
    # IO
    parser.add_argument('--checkpoint', type=str, default='mdl_model.pkl',
                       help='Path to save model (default: mdl_model.pkl)')
    parser.add_argument('--output-dir', type=str, default='mdl_analysis',
                       help='Directory to save analysis plots (default: mdl_analysis)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MDL \"Noisy Weight\" Experiment")
    print("=" * 60)
    print("Goal: Fit a sine wave while ignoring the gap in the middle.")
    
    train(args)

if __name__ == "__main__":
    main()