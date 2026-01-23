"""
Exercise 1: The "Reparameterization" Trick (Math → Code)
=========================================================

Goal: Implement the core mechanism of Bayesian Neural Networks.

Your Task:
- Implement the Gaussian sampling method
- Verify the samples match expected statistics
- Understand the "Softplus" trick

Learning Objectives:
1. How to make randomness differentiable
2. The reparameterization trick: w = μ + σ * ε
3. Why Softplus instead of Sigmoid for σ
4. Monte Carlo approximation

Time: 20-30 minutes
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt


def softplus(x):
    """
    Softplus activation: log(1 + exp(x)).
    Maps (-∞, ∞) → (0, ∞).
    Used to ensure σ > 0.
    """
    # TODO 1: Implement softplus
    # Hint: Use np.log1p(np.exp(x)) for numerical stability
    pass


def sample_gaussian(mu, rho, n_samples=10000):
    """
    Sample weights using the reparameterization trick.
    
    Args:
        mu: Mean of the distribution (the "learned value")
        rho: Unconstrained uncertainty parameter
        n_samples: Number of samples to draw
        
    Returns:
        samples: Array of sampled weights
        sigma: The computed standard deviation
        
    Formula:
        σ = Softplus(ρ)
        ε ~ N(0, 1)
        w = μ + σ * ε
        
    Why this trick?
    - Standard backprop can't differentiate through randomness
    - By reparameterizing, we move the randomness outside the graph
    - Now we can backprop through μ and ρ!
    """
    # TODO 2: Compute sigma using softplus
    # sigma = softplus(rho)
    
    # TODO 3: Sample epsilon from N(0, 1)
    # epsilon = np.random.randn(n_samples)
    
    # TODO 4: Compute w = mu + sigma * epsilon
    # samples = mu + sigma * epsilon
    
    # TODO 5: Return samples and sigma
    # return samples, sigma
    
    pass


def test_gaussian_sampling():
    """
    Test that our sampling produces correct statistics.
    """
    print("Testing Gaussian Sampling...\n")
    
    # Test 1: Basic sampling
    mu = 5.0
    rho = 0.0  # Should give sigma ≈ log(2) ≈ 0.693
    samples, sigma = sample_gaussian(mu, rho, n_samples=100000)
    
    # TODO 6: Verify the mean is close to mu
    # sample_mean = np.mean(samples)
    # assert abs(sample_mean - mu) < 0.05, f"Mean mismatch: {sample_mean} vs {mu}"
    
    # TODO 7: Verify the std is close to sigma
    # sample_std = np.std(samples)
    # assert abs(sample_std - sigma) < 0.05, f"Std mismatch: {sample_std} vs {sigma}"
    
    print("✓ Basic sampling test passed")
    
    # Test 2: Different rho values
    print("\nTesting different rho values:")
    for rho_val in [-2, 0, 2, 4]:
        # TODO 8: Sample and compute expected sigma
        # samples, sigma = sample_gaussian(5.0, rho_val, 50000)
        
        # TODO 9: Plot histogram to visualize
        # print(f"  ρ={rho_val:2d} → σ={sigma:.3f}")
        
        pass
    
    # Test 3: Visualization
    # TODO 10: Create a visualization showing different distributions
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_gaussian_sampling()
