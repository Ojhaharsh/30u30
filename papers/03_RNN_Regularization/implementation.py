"""
RNN Regularization Techniques in NumPy
======================================

A complete implementation of regularization strategies:
- Dropout
- Layer Normalization
- Weight Decay
- Early Stopping
- Validation Monitoring

This builds on Day 2's LSTM but adds defensive programming.

Author: 30u30 Project
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)


def dropout_forward(x: np.ndarray, keep_prob: float, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dropout forward pass.
    
    During training: Randomly set activations to 0 with probability (1-keep_prob)
    During testing: Don't apply dropout (use full network)
    
    Args:
        x: Activations, shape (neurons,)
        keep_prob: Probability to keep each neuron (e.g., 0.8)
        training: Whether in training mode
        
    Returns:
        x_out: Dropped out activations
        mask: Binary mask used (for backward pass)
    """
    if not training or keep_prob == 1.0:
        return x, None
    
    # Create mask: 1 with probability keep_prob, 0 otherwise
    mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
    
    return x * mask, mask


def dropout_backward(dh: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Dropout backward pass.
    
    Args:
        dh: Gradient of loss w.r.t. output
        mask: Mask from forward pass
        
    Returns:
        dh_out: Gradient w.r.t. input
    """
    if mask is None:
        return dh
    
    return dh * mask


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                       eps: float = 1e-5) -> Tuple[np.ndarray, Dict]:
    """
    Layer Normalization forward pass.
    
    Normalize each sample independently:
    1. Compute mean and variance across features
    2. Normalize to zero mean, unit variance
    3. Scale by gamma, shift by beta
    
    Args:
        x: Input, shape (batch_size, features)
        gamma: Scale parameter, shape (features,)
        beta: Shift parameter, shape (features,)
        eps: Small constant for numerical stability
        
    Returns:
        x_norm: Normalized output
        cache: Data for backward pass
    """
    # Compute statistics per sample
    mean = np.mean(x, axis=1, keepdims=True)  # (batch, 1)
    var = np.var(x, axis=1, keepdims=True)     # (batch, 1)
    
    # Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    out = gamma * x_hat + beta
    
    cache = (x, x_hat, mean, var, gamma, beta, eps)
    return out, cache


def layer_norm_backward(dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer Normalization backward pass.
    
    Args:
        dout: Gradient of loss w.r.t. output
        cache: Cache from forward pass
        
    Returns:
        dx: Gradient w.r.t. input
        dgamma: Gradient w.r.t. gamma
        dbeta: Gradient w.r.t. beta
    """
    x, x_hat, mean, var, gamma, beta, eps = cache
    
    N, D = x.shape
    
    # Gradient w.r.t. scale and shift
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Gradient w.r.t. normalized input
    dx_hat = dout * gamma
    
    # Gradient w.r.t. variance and mean
    dx_dvar = dx_hat * (x - mean) * -0.5 * (var + eps) ** -1.5
    dvar = np.sum(dx_dvar, axis=1, keepdims=True)
    
    dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=1, keepdims=True)
    dmean += dvar * np.sum(-2 * (x - mean), axis=1, keepdims=True) / D
    
    # Gradient w.r.t. input
    dx = dx_hat / np.sqrt(var + eps)
    dx += dvar * 2 * (x - mean) / D
    dx += dmean / D
    
    return dx, dgamma, dbeta


class RegularizedLSTM:
    """
    LSTM with regularization techniques.
    
    Features:
    - Dropout on inputs and hidden states
    - Layer normalization
    - Weight decay support
    - Gradient clipping
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int,
                 dropout_keep_prob: float = 0.8, use_layer_norm: bool = True):
        """
        Initialize regularized LSTM.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size
            output_size: Output size (vocab_size for LM)
            dropout_keep_prob: Dropout keep probability
            use_layer_norm: Whether to use layer normalization
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_keep_prob = dropout_keep_prob
        self.use_layer_norm = use_layer_norm
        
        # Parameters
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        # LSTM gates (same as before)
        self.Wf = np.random.randn(hidden_size, input_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size) * scale
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
        # Output layer
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))
        
        # Layer norm parameters
        if self.use_layer_norm:
            self.gamma_h = np.ones((hidden_size, 1))
            self.beta_h = np.zeros((hidden_size, 1))
    
    def forward(self, inputs: list, targets: list, h_prev: np.ndarray, 
                c_prev: np.ndarray, training: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Forward pass with regularization.
        
        Args:
            inputs: List of input indices
            targets: List of target indices
            h_prev: Previous hidden state
            c_prev: Previous cell state
            training: Whether in training mode (affects dropout)
            
        Returns:
            loss: Cross-entropy loss
            h: Final hidden state
            c: Final cell state
        """
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            # One-hot encode
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            # Dropout on input
            x_dropped, x_mask = dropout_forward(xs[t], self.dropout_keep_prob, training)
            
            # LSTM forward
            combined = np.vstack([hs[t-1], x_dropped])
            
            f = sigmoid(np.dot(self.Wf, combined) + self.bf)
            i = sigmoid(np.dot(self.Wi, combined) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
            cs[t] = f * cs[t-1] + i * c_tilde
            
            o = sigmoid(np.dot(self.Wo, combined) + self.bo)
            hs[t] = o * np.tanh(cs[t])
            
            # Dropout on hidden state
            h_dropped, h_mask = dropout_forward(hs[t], self.dropout_keep_prob, training)
            
            # Layer norm on hidden state
            if self.use_layer_norm:
                hs[t], _ = layer_norm_forward(h_dropped.T, self.gamma_h.T, self.beta_h.T)
                hs[t] = hs[t].T
            
            # Output layer
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = softmax(ys[t])
            
            # Loss
            loss += -np.log(ps[t][target, 0] + 1e-8)
        
        return loss, hs[len(inputs)-1], cs[len(inputs)-1]
    
    def parameter_count(self) -> int:
        """Count total parameters."""
        count = (self.Wf.size + self.Wi.size + self.Wc.size + self.Wo.size +
                self.bf.size + self.bi.size + self.bc.size + self.bo.size +
                self.Why.size + self.by.size)
        
        if self.use_layer_norm:
            count += self.gamma_h.size + self.beta_h.size
        
        return count


class EarlyStoppingMonitor:
    """
    Monitor training and perform early stopping.
    
    Tracks validation loss and saves best model when it improves.
    Stops when validation loss doesn't improve for 'patience' epochs.
    """
    
    def __init__(self, patience: int = 5, verbose: bool = True):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            verbose: Whether to print status
        """
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.best_model_state = None
    
    def check(self, val_loss: float, epoch: int, model_state: Optional[Dict] = None) -> bool:
        """
        Check if should continue training.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch
            model_state: Model parameters to save
            
        Returns:
            True if should continue, False if should stop
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.no_improve_count = 0
            self.best_model_state = model_state
            
            if self.verbose:
                print(f"  [ok] Epoch {epoch}: val_loss improved to {val_loss:.4f}")
        else:
            self.no_improve_count += 1
            
            if self.verbose:
                print(f"  [FAIL] Epoch {epoch}: val_loss {val_loss:.4f} "
                      f"(no improve for {self.no_improve_count}/{self.patience})")
        
        return self.no_improve_count < self.patience
    
    def should_stop(self) -> bool:
        """Check if should stop training."""
        return self.no_improve_count >= self.patience


class RegularizationConfig:
    """Configuration for regularization strategies."""
    
    def __init__(self):
        self.dropout_keep_prob = 0.8
        self.weight_decay = 0.0001
        self.gradient_clip = 5.0
        self.use_layer_norm = True
        self.learning_rate = 0.01
        self.batch_size = 32
        self.val_split = 0.1
        self.patience = 5


# ============================================================================
# UTILITIES
# ============================================================================

def compute_l2_penalty(weights_list: list, weight_decay: float) -> float:
    """
    Compute L2 regularization penalty.
    
    L2_penalty = weight_decay * sum(w^2)
    
    Args:
        weights_list: List of weight matrices
        weight_decay: Coefficient (lambda)
        
    Returns:
        L2 penalty (scalar)
    """
    penalty = 0.0
    for w in weights_list:
        penalty += np.sum(w ** 2)
    
    return weight_decay * penalty / 2


def regularized_loss(model_loss: float, weights_list: list, 
                     weight_decay: float) -> float:
    """
    Compute total loss with L2 regularization.
    
    Total_loss = model_loss + L2_penalty
    
    Args:
        model_loss: Loss from model (e.g., cross-entropy)
        weights_list: List of weight matrices
        weight_decay: Weight decay coefficient
        
    Returns:
        Total loss
    """
    l2_penalty = compute_l2_penalty(weights_list, weight_decay)
    return model_loss + l2_penalty


def clip_gradients(grads_dict: Dict, clip_value: float) -> Dict:
    """
    Clip gradients to prevent explosion.
    
    Clips all gradient magnitudes to [-clip_value, clip_value].
    
    Args:
        grads_dict: Dictionary of gradients
        clip_value: Clipping threshold
        
    Returns:
        Clipped gradients dictionary
    """
    clipped = {}
    for key, grad in grads_dict.items():
        clipped[key] = np.clip(grad, -clip_value, clip_value)
    return clipped


if __name__ == "__main__":
    print("Regularization techniques module loaded.")
    print("Classes: RegularizedLSTM, EarlyStoppingMonitor")
    print("Functions: dropout_forward/backward, layer_norm_forward/backward")
