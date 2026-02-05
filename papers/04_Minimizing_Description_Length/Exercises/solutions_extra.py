"""
Solutions for Day 4 Exercises
=============================

Spoiler Alert! Try to implement these yourself first.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation import MDLNetwork

def exercise_1_reparameterization():
    print("\n--- Exercise 1: The Reparameterization Trick ---")
    
    def sample_gaussian(mu, rho, n_samples=10000):
        # 1. Softplus transform
        sigma = np.log1p(np.exp(rho))
        
        # 2. Sample epsilon
        epsilon = np.random.randn(n_samples)
        
        # 3. Scale and shift
        w = mu + sigma * epsilon
        return w, sigma

    # Test case
    mu_target = 5.0
    rho_target = -2.0 # sigma should be approx 0.126
    
    samples, expected_sigma = sample_gaussian(mu_target, rho_target)
    
    print(f"Target Mean: {mu_target}, Measured Mean: {np.mean(samples):.4f}")
    print(f"Target Std:  {expected_sigma:.4f}, Measured Std:  {np.std(samples):.4f}")
    
    if np.abs(np.mean(samples) - mu_target) < 0.1:
        print("[OK] Mean check passed")
    if np.abs(np.std(samples) - expected_sigma) < 0.1:
        print("[OK] Sigma check passed")

def exercise_2_gap_experiment():
    print("\n--- Exercise 2: The Gap Experiment ---")
    
    # 1. Generate Cubic Data with Gap
    X_left = np.linspace(-4, -2, 50).reshape(-1, 1)
    X_right = np.linspace(2, 4, 50).reshape(-1, 1)
    X_train = np.concatenate([X_left, X_right])
    
    # Normalize y to keep gradients stable
    y_train = (X_train**3) / 10.0 + np.random.randn(100, 1) * 0.1
    
    # 2. Train
    net = MDLNetwork(1, 20, 1)
    lr = 0.01
    kl_weight = 0.1
    
    print("Training on cubic gap (this takes a moment)...")
    for i in range(2000):
        preds = net.forward(X_train)
        mse = np.mean((preds - y_train)**2)
        kl = net.total_kl() / 100.0
        
        loss = mse + kl_weight * kl
        
        net.backward(2*(preds - y_train)/100.0)
        net.update_weights(lr, kl_weight/100.0)
    
    # 3. Visualize
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    preds = np.array([net.forward(X_test) for _ in range(100)]).squeeze()
    
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c='red', label='Data')
    plt.plot(X_test, mu, label='Mean Pred')
    plt.fill_between(X_test.flatten(), mu-2*sigma, mu+2*sigma, alpha=0.3, label='Uncertainty')
    plt.title("Exercise 2: Cubic Function with Gap")
    plt.legend()
    plt.show()
    print("[OK] Check the plot: Uncertainty should expand in [-2, 2]")

def exercise_3_beta_sensitivity():
    print("\n--- Exercise 3: Beta (KL Weight) Sensitivity ---")
    print("This is a thought experiment code block.")
    
    betas = [0.0, 0.1, 100.0]
    descriptions = [
        "Overfitting (Standard NN behavior)",
        "Balanced (Good Uncertainty)",
        "Underfitting (Prior Collapse - Flat line)"
    ]
    
    for b, desc in zip(betas, descriptions):
        print(f"Beta = {b}: {desc}")
        # To run this, you would loop Exercise 2 with different kl_weight values

def exercise_4_monte_carlo():
    print("\n--- Exercise 4: Monte Carlo Smoothing ---")
    
    # Setup dummy network
    net = MDLNetwork(1, 10, 1)
    x = np.array([[0.5]])
    
    print("Predicting 10 times...")
    preds = [net.forward(x).item() for _ in range(10)]
    print(f"Individual Predictions: {[round(p, 3) for p in preds]}")
    print(f"Variance: {np.var(preds):.5f}")
    print("[OK] Notice how single predictions jitter? That's why we average.")

def exercise_5_pruning():
    print("\n--- Exercise 5: The Free Lunch (Pruning) ---")
    
    net = MDLNetwork(1, 20, 1)
    # Simulate training by setting some weights to have high uncertainty
    net.layer1.w_rho += 5.0 # Make them very uncertain
    
    # Calculate SNR
    mu = net.layer1.w_mu
    # sigma approx log(1 + exp(rho))
    sigma = np.log1p(np.exp(net.layer1.w_rho))
    
    snr = np.abs(mu) / sigma
    
    # Prune
    threshold = 0.1
    mask = snr < threshold
    n_pruned = np.sum(mask)
    total = mu.size
    
    print(f"Total Weights: {total}")
    print(f"Pruned Weights (SNR < {threshold}): {n_pruned}")
    print(f"Compression Rate: {n_pruned/total*100:.1f}%")
    print("[OK] These pruned weights can be set to 0 without affecting the output mean significantly.")

if __name__ == "__main__":
    # Uncomment to run
    exercise_1_reparameterization()
    # exercise_2_gap_experiment() # Requires matplotlib
    exercise_4_monte_carlo()
    exercise_5_pruning()