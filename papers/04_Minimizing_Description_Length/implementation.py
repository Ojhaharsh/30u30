"""
MDL (Minimum Description Length) / Bayesian Neural Network Implementation
=========================================================================

A complete, educational implementation of a "Noisy Weight" Neural Network.

This is fundamentally different from the Standard RNN (Day 1) or LSTM (Day 2).
Instead of learning a single fixed value for every weight (e.g., w = 0.5), 
we learn a PROBABILITY DISTRIBUTION for every weight (e.g., w ~ Gaussian(0.5, 0.1)).

We have:
- Learnable Means (mu)
- Learnable Variances (sigma) via the parameter 'rho'
- The "Reparameterization Trick" to allow backpropagation through randomness
- A Loss function that combines Error (NLL) + Complexity (KL Divergence)

This implements the core idea of Hinton & Van Camp (1993):
"Simpler weights (more noise/uncertainty) = Better Compression = Better Generalization"

Author: 30u30 Project
License: MIT
"""

import numpy as np
import pickle


def softplus(x):
    """
    Softplus activation: log(1 + exp(x)).
    
    Used to transform the raw parameter 'rho' into a positive standard deviation 'sigma'.
    We need sigma > 0, and softplus is a smooth approximation of ReLU that stays positive.
    
    Args:
        x: Input array (rho)
        
    Returns:
        Array of positive values (sigma)
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    """Sigmoid activation for the derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """ReLU activation for hidden layers."""
    return np.maximum(0, x)


class BayesianLinear:
    """
    A single Dense Layer where weights are probability distributions.
    
    Instead of w, we have:
    - w_mu: The mean of the weight
    - w_rho: Controls the variance (sigma = softplus(w_rho))
    
    Forward pass:
    1. Calculate sigma from rho
    2. Sample noise (epsilon) ~ N(0, 1)
    3. Construct weight: w = mu + sigma * epsilon
    4. Compute output: y = x @ w + b
    """
    
    def __init__(self, input_size, output_size, init_scale=0.1):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize Means (mu) randomly (like standard weights)
        # Using He initialization style
        scale = np.sqrt(2.0 / input_size)
        self.w_mu = np.random.randn(input_size, output_size) * scale
        self.b_mu = np.zeros(output_size)
        
        # Initialize Rho (Uncertainty control)
        # We start with a small negative value (e.g., -3) so that 
        # sigma = softplus(-3) is small (~0.05). 
        # We want the model to start slightly uncertain, but not purely random.
        self.w_rho = np.ones((input_size, output_size)) * -3.0
        self.b_rho = np.ones(output_size) * -3.0
        
        # Gradients storage
        self.grads = {}
        
        # Cache for backprop
        self.cache = None

    def sample_weights(self):
        """
        Sample actual weights from the learned distributions.
        This is the "Bits Back" / Reparameterization Step.
        """
        # 1. Convert rho to sigma (standard deviation)
        # sigma = log(1 + exp(rho))
        self.w_sigma = softplus(self.w_rho)
        self.b_sigma = softplus(self.b_rho)
        
        # 2. Sample noise (epsilon) from standard normal N(0,1)
        self.w_epsilon = np.random.randn(self.input_size, self.output_size)
        self.b_epsilon = np.random.randn(self.output_size)
        
        # 3. Construct the noisy weights
        # w = mu + sigma * epsilon
        self.w_sample = self.w_mu + self.w_sigma * self.w_epsilon
        self.b_sample = self.b_mu + self.b_sigma * self.b_epsilon
        
        return self.w_sample, self.b_sample

    def forward(self, x):
        """
        Forward pass using sampled weights.
        """
        # Sample weights for this specific forward pass
        w, b = self.sample_weights()
        
        # Linear transformation: y = xW + b
        out = x @ w + b
        
        # Cache for backward pass
        self.cache = (x, w, b)
        return out

    def kl_divergence(self):
        """
        Compute the Complexity Cost (KL Divergence) for this layer.
        
        We calculate KL( q(w) || p(w) )
        - q(w): The learned posterior N(mu, sigma)
        - p(w): The prior N(0, 1) (Standard Gaussian)
        
        Formula for KL between N(mu, sigma) and N(0, 1):
        KL = 0.5 * sum( sigma^2 + mu^2 - 1 - log(sigma^2) )
        """
        # KL for weights
        kl_w = 0.5 * np.sum(
            self.w_sigma**2 + self.w_mu**2 - 1 - 2*np.log(self.w_sigma + 1e-8)
        )
        
        # KL for biases
        kl_b = 0.5 * np.sum(
            self.b_sigma**2 + self.b_mu**2 - 1 - 2*np.log(self.b_sigma + 1e-8)
        )
        
        return kl_w + kl_b

    def backward(self, d_out):
        """
        Compute gradients for mu and rho.
        
        This is tricky! We have two sources of gradients:
        1. The Data Loss (NLL): flows back through the sampled weights
        2. The Complexity Loss (KL): flows directly from mu and sigma
        
        Args:
            d_out: Gradient of the Data Loss w.r.t the output of this layer (dL/dy)
        
        Returns:
            d_x: Gradient w.r.t input x (to pass to previous layer)
        """
        x, _, _ = self.cache
        
        # === 1. Gradients from Data Loss (Error Cost) ===
        # We derived these via the Chain Rule on the Reparameterization Trick.
        
        # Gradient w.r.t sampled weights (Standard Backprop)
        # dL/dw = x.T @ dL/dy
        d_w_sample = x.T @ d_out
        d_b_sample = np.sum(d_out, axis=0)
        
        # Gradient w.r.t Mean (mu)
        # dL/dmu = dL/dw * dw/dmu = dL/dw * 1
        d_nll_w_mu = d_w_sample
        d_nll_b_mu = d_b_sample
        
        # Gradient w.r.t Rho
        # dL/drho = dL/dw * dw/dsigma * dsigma/drho
        # dw/dsigma = epsilon
        # dsigma/drho = sigmoid(rho)
        d_nll_w_rho = d_w_sample * self.w_epsilon * sigmoid(self.w_rho)
        d_nll_b_rho = d_b_sample * self.b_epsilon * sigmoid(self.b_rho)
        
        # === 2. Gradients from KL Divergence (Complexity Cost) ===
        # KL = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
        
        # Gradient w.r.t Mean (mu)
        # dKL/dmu = mu
        d_kl_w_mu = self.w_mu
        d_kl_b_mu = self.b_mu
        
        # Gradient w.r.t Rho
        # dKL/drho = dKL/dsigma * dsigma/drho
        # dKL/dsigma = sigma - 1/sigma
        d_kl_w_sigma = self.w_sigma - (1.0 / (self.w_sigma + 1e-8))
        d_kl_b_sigma = self.b_sigma - (1.0 / (self.b_sigma + 1e-8))
        
        d_kl_w_rho = d_kl_w_sigma * sigmoid(self.w_rho)
        d_kl_b_rho = d_kl_b_sigma * sigmoid(self.b_rho)
        
        # === 3. Combine and Store ===
        # The optimizer usually handles the weighting (beta), but we store raw gradients here.
        # We will separate NLL and KL gradients so the training loop can weight them.
        
        self.grads = {
            'nll_w_mu': d_nll_w_mu, 'kl_w_mu': d_kl_w_mu,
            'nll_b_mu': d_nll_b_mu, 'kl_b_mu': d_kl_b_mu,
            'nll_w_rho': d_nll_w_rho, 'kl_w_rho': d_kl_w_rho,
            'nll_b_rho': d_nll_b_rho, 'kl_b_rho': d_kl_b_rho
        }
        
        # Gradient w.r.t input x (to pass down)
        d_x = d_out @ self.w_sample.T
        return d_x


class MDLNetwork:
    """
    A simple 2-Layer Bayesian Neural Network.
    Structure: Input -> BayesianLinear -> ReLU -> BayesianLinear -> Output
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create layers
        self.layer1 = BayesianLinear(input_size, hidden_size)
        self.layer2 = BayesianLinear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass.
        Note: Because weights are sampled, calling this twice on same x
        will produce slightly different results! This is the "Uncertainty".
        """
        self.h = relu(self.layer1.forward(x))
        out = self.layer2.forward(self.h)
        return out
    
    def total_kl(self):
        """Total Complexity Cost (KL) of the entire network."""
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()
    
    def backward(self, d_loss):
        """
        Backward pass.
        Args:
            d_loss: Gradient of the Data Loss (MSE/NLL) w.r.t network output
        """
        # Backprop through layer 2
        d_h = self.layer2.backward(d_loss)
        
        # Backprop through ReLU
        d_h[self.h <= 0] = 0
        
        # Backprop through layer 1
        self.layer1.backward(d_h)
        
    def update_weights(self, learning_rate, kl_weight):
        """
        Update parameters using SGD.
        
        Total Gradient = Data_Grad + (kl_weight * KL_Grad)
        """
        for layer in [self.layer1, self.layer2]:
            # Update Mu (Means)
            layer.w_mu -= learning_rate * (layer.grads['nll_w_mu'] + kl_weight * layer.grads['kl_w_mu'])
            layer.b_mu -= learning_rate * (layer.grads['nll_b_mu'] + kl_weight * layer.grads['kl_b_mu'])
            
            # Update Rho (Variances)
            layer.w_rho -= learning_rate * (layer.grads['nll_w_rho'] + kl_weight * layer.grads['kl_w_rho'])
            layer.b_rho -= learning_rate * (layer.grads['nll_b_rho'] + kl_weight * layer.grads['kl_b_rho'])

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Run the model N times to get a distribution of predictions.
        
        Returns:
            mean_pred: The average prediction
            std_pred: The standard deviation (Uncertainty)
        """
        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(x))
            
        preds = np.array(preds) # Shape: (n_samples, batch_size, output_size)
        
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)
        
        return mean_pred, std_pred

    def save(self, filepath):
        """Save model parameters."""
        params = {
            'l1_w_mu': self.layer1.w_mu, 'l1_w_rho': self.layer1.w_rho,
            'l1_b_mu': self.layer1.b_mu, 'l1_b_rho': self.layer1.b_rho,
            'l2_w_mu': self.layer2.w_mu, 'l2_w_rho': self.layer2.w_rho,
            'l2_b_mu': self.layer2.b_mu, 'l2_b_rho': self.layer2.b_rho,
            'dims': (self.input_size, self.hidden_size, self.output_size)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model."""
        with open(filepath, 'rb') as f:
            p = pickle.load(f)
            
        model = cls(*p['dims'])
        
        model.layer1.w_mu, model.layer1.w_rho = p['l1_w_mu'], p['l1_w_rho']
        model.layer1.b_mu, model.layer1.b_rho = p['l1_b_mu'], p['l1_b_rho']
        model.layer2.w_mu, model.layer2.w_rho = p['l2_w_mu'], p['l2_w_rho']
        model.layer2.b_mu, model.layer2.b_rho = p['l2_b_mu'], p['l2_b_rho']
        
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    print("MDL / Bayesian Network Implementation")
    print("=" * 60)
    print("\nThis file contains the BayesianLinear and MDLNetwork classes.")
    print("It implements 'Noisy Weights' using the reparameterization trick.")
    print("\nKey Components:")
    print("  - BayesianLinear: A layer where weights are N(mu, sigma)")
    print("  - sample_weights(): Adds noise for the 'Bits Back' step")
    print("  - kl_divergence(): Calculates the complexity cost")
    print("\nRun 'train_mdl.py' to see it learn!")