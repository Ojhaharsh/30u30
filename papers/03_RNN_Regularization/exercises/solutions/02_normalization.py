"""
Solution 2: Layer Normalization Implementation
===============================================

Complete solution with explanations for each step.
"""

import numpy as np
from typing import Tuple


class LayerNorm:
    """
    Layer Normalization for RNNs.
    
    Key Insight: Unlike BatchNorm (normalizes across batch),
    LayerNorm normalizes across features within each sample.
    This makes it perfect for RNNs where batch statistics are unreliable.
    """
    
    def __init__(self, hidden_size: int, epsilon: float = 1e-5):
        """
        SOLUTION 1: Initialize layer norm parameters
        
        Args:
            hidden_size: Size of the features to normalize
            epsilon: Small constant for numerical stability
            
        Learnable parameters:
            gamma: Scale parameter (initialized to 1)
            beta: Shift parameter (initialized to 0)
            
        Why learnable gamma/beta?
        - After normalization, all features have mean=0, std=1
        - Network might need different scales for different features
        - gamma and beta let the network learn the optimal scale/shift
        """
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(hidden_size)   # Scale
        self.beta = np.zeros(hidden_size)   # Shift
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        SOLUTION 2: Forward pass
        
        Formula:
            μ = mean(x, axis=-1)        # Mean across features
            σ² = var(x, axis=-1)        # Variance across features
            x_hat = (x - μ) / √(σ² + ε) # Normalize
            y = γ * x_hat + β           # Scale and shift
            
        Args:
            x: Input tensor (batch_size, hidden_size) or (hidden_size,)
            
        Returns:
            y: Normalized output, same shape as x
        """
        # SOLUTION 2a: Compute mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # SOLUTION 2b: Normalize
        std = np.sqrt(var + self.epsilon)
        x_hat = (x - mean) / std
        
        # SOLUTION 2c: Scale and shift
        y = self.gamma * x_hat + self.beta
        
        # SOLUTION 2d: Cache for backward pass
        self.cache = (x, x_hat, mean, std)
        
        return y
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SOLUTION 3: Backward pass
        
        This is the tricky part! We need gradients for:
        - dx: gradient w.r.t. input
        - dgamma: gradient w.r.t. scale parameter
        - dbeta: gradient w.r.t. shift parameter
        
        The math:
        - dbeta = sum(dout)
        - dgamma = sum(dout * x_hat)
        - dx = complex chain rule through normalization
        """
        x, x_hat, mean, std = self.cache
        
        # SOLUTION 3a: Gradient for beta (simple sum)
        dbeta = np.sum(dout, axis=0) if dout.ndim > 1 else dout.copy()
        
        # SOLUTION 3b: Gradient for gamma
        dgamma = np.sum(dout * x_hat, axis=0) if dout.ndim > 1 else (dout * x_hat)
        
        # SOLUTION 3c: Gradient for x (the complex part)
        N = x.shape[-1]
        
        dx_hat = dout * self.gamma
        
        # Gradient through variance
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        
        # Gradient through mean
        dmean = np.sum(dx_hat * -1 / std, axis=-1, keepdims=True)
        dmean += dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        
        # Final gradient
        dx = dx_hat / std
        dx += dvar * 2 * (x - mean) / N
        dx += dmean / N
        
        return dx, dgamma, dbeta


class LayerNormRNN:
    """
    SOLUTION 4: RNN with Layer Normalization
    
    Layer norm is applied AFTER the linear transformation,
    BEFORE the nonlinearity (tanh).
    
    Standard RNN: h = tanh(Wx + Uh + b)
    With LayerNorm: h = tanh(LayerNorm(Wx + Uh + b))
    
    Why before tanh?
    - Normalizing keeps values in good range for tanh
    - Prevents saturation (values too close to -1 or 1)
    - Gradients flow better through unsaturated tanh
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        SOLUTION 5: Initialize RNN with layer norm
        """
        self.hidden_size = hidden_size
        
        # Weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_xh = np.random.randn(hidden_size, input_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        
        # SOLUTION 5a: Layer norm for hidden state
        self.layer_norm = LayerNorm(hidden_size)
    
    def forward(self, x_sequence: np.ndarray) -> Tuple[list, list]:
        """
        SOLUTION 6: Forward pass with layer norm
        
        For each timestep:
        1. Compute linear combination: z = Wx + Uh + b
        2. Apply layer norm: z_norm = LayerNorm(z)
        3. Apply nonlinearity: h = tanh(z_norm)
        """
        seq_len, batch_size, _ = x_sequence.shape
        h = np.zeros((batch_size, self.hidden_size))
        
        hidden_states = [h]
        pre_activations = []  # Before tanh, for backprop
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            
            # SOLUTION 6a: Linear transformation
            z = (self.W_xh @ x_t.T + self.W_hh @ h.T + self.b_h.reshape(-1, 1)).T
            
            # SOLUTION 6b: Apply layer norm
            z_norm = self.layer_norm.forward(z)
            
            # SOLUTION 6c: Apply tanh
            h = np.tanh(z_norm)
            
            pre_activations.append(z_norm)
            hidden_states.append(h)
        
        return hidden_states, pre_activations


def numerical_gradient_check():
    """
    SOLUTION 7: Verify gradients numerically
    
    For any function f, the numerical gradient is:
        df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)
    
    We compare this to our analytical gradient from backward().
    """
    print("Running numerical gradient check...")
    
    ln = LayerNorm(hidden_size=5)
    np.random.seed(42)
    
    x = np.random.randn(3, 5)  # (batch, features)
    
    # Forward and backward
    y = ln.forward(x)
    dout = np.random.randn(*y.shape)
    dx, dgamma, dbeta = ln.backward(dout)
    
    # Numerical gradient for gamma
    epsilon = 1e-5
    num_dgamma = np.zeros_like(ln.gamma)
    
    for i in range(len(ln.gamma)):
        # f(gamma + epsilon)
        ln.gamma[i] += epsilon
        y_plus = ln.forward(x)
        loss_plus = np.sum(y_plus * dout)
        ln.gamma[i] -= epsilon
        
        # f(gamma - epsilon)
        ln.gamma[i] -= epsilon
        y_minus = ln.forward(x)
        loss_minus = np.sum(y_minus * dout)
        ln.gamma[i] += epsilon
        
        num_dgamma[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare
    diff = np.max(np.abs(dgamma - num_dgamma))
    print(f"  Max difference (gamma): {diff:.2e}")
    assert diff < 1e-4, f"Gradient check failed! Diff: {diff}"
    print("  ✓ Gamma gradient correct")
    
    print("✓ Numerical gradient check passed!\n")


def test_layer_norm_basic():
    """Test basic layer norm functionality."""
    print("Testing basic layer norm...")
    
    ln = LayerNorm(hidden_size=10)
    
    # Test single sample
    x = np.random.randn(10) * 5 + 3  # Arbitrary mean and scale
    y = ln.forward(x)
    
    # Should have mean ≈ 0 and std ≈ 1
    assert abs(np.mean(y)) < 0.01, f"Mean should be ~0, got {np.mean(y)}"
    assert abs(np.std(y) - 1) < 0.01, f"Std should be ~1, got {np.std(y)}"
    print("  ✓ Single sample normalized correctly")
    
    # Test batch
    x_batch = np.random.randn(5, 10) * 10 + 7
    y_batch = ln.forward(x_batch)
    
    # Each sample should be normalized independently
    for i in range(5):
        mean = np.mean(y_batch[i])
        std = np.std(y_batch[i])
        assert abs(mean) < 0.01, f"Sample {i} mean should be ~0"
        assert abs(std - 1) < 0.01, f"Sample {i} std should be ~1"
    print("  ✓ Batch normalized correctly (each sample independent)")
    
    print("✓ Basic layer norm tests passed!\n")


def test_rnn_layer_norm():
    """Test RNN with layer norm."""
    print("Testing RNN with layer norm...")
    
    rnn = LayerNormRNN(input_size=5, hidden_size=10)
    
    # Create sequence
    seq = np.random.randn(4, 2, 5)  # (seq_len, batch, input)
    
    hidden_states, _ = rnn.forward(seq)
    
    assert len(hidden_states) == 5, f"Expected 5 states, got {len(hidden_states)}"
    assert hidden_states[-1].shape == (2, 10), f"Wrong shape: {hidden_states[-1].shape}"
    
    print(f"  ✓ Processed sequence of length 4")
    print(f"  ✓ Final hidden state shape: {hidden_states[-1].shape}")
    
    print("✓ RNN layer norm tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Solution 2: Layer Normalization Implementation")
    print("="*60 + "\n")
    
    test_layer_norm_basic()
    numerical_gradient_check()
    test_rnn_layer_norm()
    
    print("🎉 All tests passed!")
