"""
Dropout: A Simple Way to Prevent Neural Networks from Overfitting

Complete implementation of Dropout and variants for educational purposes.
Every function is heavily commented to explain the concepts.

Based on: Srivastava et al. (2014)
Paper: https://jmlr.org/papers/v15/srivastava14a.html

Author: 30u30 Project
Date: 2024
"""

import numpy as np
from typing import Tuple, Optional, List, Union
import warnings


# =============================================================================
# CORE DROPOUT IMPLEMENTATIONS
# =============================================================================

class Dropout:
    """
    Standard Dropout Layer.
    
    During training, randomly sets elements to zero with probability (1-p),
    and scales the remaining elements by 1/p (inverted dropout).
    
    During inference, the layer is a no-op.
    
    Args:
        p: Probability of KEEPING an element (not dropping!).
           p=0.5 means 50% of neurons are kept.
           
    Note:
        PyTorch's nn.Dropout uses p as the DROP probability, which is opposite!
        We follow the original paper's convention where p = keep probability.
    
    Example:
        >>> dropout = Dropout(p=0.5)
        >>> dropout.training = True
        >>> x = np.ones((3, 4))
        >>> y = dropout.forward(x)  # About half will be 0, rest scaled by 2
    """
    
    def __init__(self, p: float = 0.5):
        if not 0 < p <= 1:
            raise ValueError(f"Keep probability must be in (0, 1], got {p}")
        
        self.p = p  # Keep probability
        self.mask = None  # Cached for backward pass
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with dropout.
        
        Args:
            x: Input array of any shape
            
        Returns:
            Dropped and scaled array (same shape as input)
        """
        if not self.training:
            # Inference: return unchanged (inverted dropout scaling already applied)
            return x
        
        # Generate Bernoulli mask: 1 with probability p, 0 with probability (1-p)
        self.mask = (np.random.rand(*x.shape) < self.p).astype(np.float32)
        
        # Apply mask and scale by 1/p (inverted dropout)
        # This ensures expected value remains the same: E[x * mask / p] = x
        output = x * self.mask / self.p
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through dropout.
        
        Gradients only flow through neurons that were kept.
        Same scaling factor is applied.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with dropout pattern applied
        """
        if not self.training:
            return grad_output
        
        # Apply same mask and scaling to gradients
        return grad_output * self.mask / self.p
    
    def train(self):
        """Set to training mode (dropout enabled)."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode (dropout disabled)."""
        self.training = False
    
    def __repr__(self):
        return f"Dropout(p={self.p}, training={self.training})"


class NaiveDropout:
    """
    Original (non-inverted) Dropout implementation.
    
    During training: Apply mask without scaling.
    During inference: Scale output by p.
    
    This is the original formulation from the paper, but inverted dropout
    (above) is preferred in practice because:
    1. No need to modify inference code
    2. Faster inference (no multiplication by p)
    3. Less error-prone (easy to forget test-time scaling)
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            # Must scale at test time!
            return x * self.p
        
        self.mask = (np.random.rand(*x.shape) < self.p).astype(np.float32)
        return x * self.mask  # No scaling during training
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad_output * self.p
        return grad_output * self.mask


class Dropout2D:
    """
    Spatial Dropout for 2D feature maps (CNNs).
    
    Instead of dropping individual pixels, drops entire channels.
    This is more appropriate for convolutional layers because nearby
    pixels are highly correlated.
    
    Args:
        p: Probability of keeping each channel
        
    Shape:
        Input: (batch_size, channels, height, width)
        Output: Same shape as input
        
    Note:
        A single mask is generated per channel and broadcast across spatial dims.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with spatial dropout.
        
        Args:
            x: Input of shape (batch, channels, height, width)
        """
        if not self.training:
            return x
        
        batch_size, channels, height, width = x.shape
        
        # Create mask for each channel (not each pixel)
        channel_mask = (np.random.rand(batch_size, channels, 1, 1) < self.p)
        
        # Broadcast to full spatial dimensions
        self.mask = np.broadcast_to(channel_mask, x.shape).astype(np.float32)
        
        return x * self.mask / self.p
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad_output
        return grad_output * self.mask / self.p
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


class Dropout1D:
    """
    Spatial Dropout for 1D sequences.
    
    Drops entire channels for sequence data.
    
    Shape:
        Input: (batch_size, channels, sequence_length)
        Output: Same shape as input
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x
        
        batch_size, channels, seq_len = x.shape
        
        # One mask value per channel
        channel_mask = (np.random.rand(batch_size, channels, 1) < self.p)
        self.mask = np.broadcast_to(channel_mask, x.shape).astype(np.float32)
        
        return x * self.mask / self.p
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad_output
        return grad_output * self.mask / self.p
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


class AlphaDropout:
    """
    Alpha Dropout for Self-Normalizing Neural Networks (SELU activation).
    
    Unlike standard dropout which zeros values, Alpha Dropout maintains
    zero mean and unit variance after dropout.
    
    Used with SELU activation for self-normalizing networks.
    
    Reference: Klambauer et al. (2017) - "Self-Normalizing Neural Networks"
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True
        
        # Constants for alpha dropout with SELU
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        
        # Compute the affine coefficients
        alpha_p = -self.alpha * self.scale
        
        # Mean and variance of the dropped units
        self.a = ((1 - self.p) * (1 + self.p * alpha_p ** 2)) ** (-0.5)
        self.b = -self.a * alpha_p * (1 - self.p)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x
        
        self.mask = (np.random.rand(*x.shape) < self.p).astype(np.float32)
        
        # Set dropped values to alpha * scale (not zero!)
        alpha_scale = -self.alpha * self.scale
        output = np.where(self.mask, x, alpha_scale)
        
        # Rescale to maintain mean and variance
        output = self.a * output + self.b
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad_output
        return grad_output * self.mask * self.a
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


# =============================================================================
# ADVANCED DROPOUT VARIANTS
# =============================================================================

class DropConnect:
    """
    DropConnect: Drop weights instead of activations.
    
    Instead of dropping neurons, DropConnect randomly drops individual
    connections (weights). This provides finer-grained regularization.
    
    Reference: Wan et al. (2013) - "Regularization of Neural Networks using DropConnect"
    
    Note: This is applied to a linear layer's weights, not to activations.
    """
    
    def __init__(self, in_features: int, out_features: int, p: float = 0.5):
        self.p = p
        self.training = True
        
        # Initialize weights
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        
        # Cached values for backward
        self.mask = None
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with DropConnect.
        
        Args:
            x: Input of shape (batch_size, in_features)
        """
        self.input = x
        
        if not self.training:
            return x @ self.W.T + self.b
        
        # Drop weights
        self.mask = (np.random.rand(*self.W.shape) < self.p).astype(np.float32)
        W_dropped = self.W * self.mask / self.p
        
        return x @ W_dropped.T + self.b
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (grad_input, grad_W, grad_b)"""
        if not self.training:
            W_effective = self.W
        else:
            W_effective = self.W * self.mask / self.p
        
        grad_input = grad_output @ W_effective
        grad_W = grad_output.T @ self.input
        grad_b = grad_output.sum(axis=0)
        
        return grad_input, grad_W, grad_b


class DropBlock:
    """
    DropBlock: Structured dropout for convolutional networks.
    
    Drops contiguous regions of feature maps instead of random elements.
    More effective than standard dropout for CNNs because it forces
    the network to look at larger context.
    
    Reference: Ghiasi et al. (2018) - "DropBlock: A regularization technique for CNNs"
    
    Args:
        block_size: Size of the square block to drop
        p: Probability of dropping (gamma in the paper)
    """
    
    def __init__(self, block_size: int = 7, p: float = 0.1):
        self.block_size = block_size
        self.p = p  # Drop probability
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with DropBlock.
        
        Args:
            x: Input of shape (batch, channels, height, width)
        """
        if not self.training:
            return x
        
        batch, channels, height, width = x.shape
        
        # Compute gamma (probability of dropping a seed point)
        gamma = self._compute_gamma(height, width)
        
        # Generate seed mask (centers of blocks to drop)
        seed_mask = np.random.rand(batch, channels, height, width) < gamma
        
        # Expand seeds to blocks
        self.mask = self._create_block_mask(seed_mask)
        
        # Apply mask
        keep_prob = self.mask.sum() / self.mask.size
        if keep_prob < 0.01:
            keep_prob = 0.01  # Avoid division by very small number
        
        return x * self.mask / keep_prob
    
    def _compute_gamma(self, height: int, width: int) -> float:
        """Compute the probability of setting a seed point."""
        # Formula from the paper
        feat_area = height * width
        block_area = self.block_size ** 2
        valid_area = (height - self.block_size + 1) * (width - self.block_size + 1)
        
        gamma = (self.p * feat_area) / (block_area * valid_area)
        return min(gamma, 1.0)
    
    def _create_block_mask(self, seed_mask: np.ndarray) -> np.ndarray:
        """Expand seed points to blocks."""
        batch, channels, height, width = seed_mask.shape
        
        # Start with keep mask
        mask = np.ones_like(seed_mask, dtype=np.float32)
        
        # For each seed point, zero out a block
        half_block = self.block_size // 2
        
        for b in range(batch):
            for c in range(channels):
                seeds = np.argwhere(seed_mask[b, c])
                for (y, x) in seeds:
                    y_start = max(0, y - half_block)
                    y_end = min(height, y + half_block + 1)
                    x_start = max(0, x - half_block)
                    x_end = min(width, x + half_block + 1)
                    mask[b, c, y_start:y_end, x_start:x_end] = 0
        
        return mask
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad_output
        
        keep_prob = self.mask.sum() / self.mask.size
        if keep_prob < 0.01:
            keep_prob = 0.01
        
        return grad_output * self.mask / keep_prob


class ScheduledDropout:
    """
    Dropout with scheduled rate that changes during training.
    
    Can be used to:
    - Start with no dropout and gradually increase
    - Start with high dropout and decrease (curriculum learning)
    """
    
    def __init__(self, 
                 initial_p: float = 0.5, 
                 final_p: float = 0.5, 
                 num_steps: int = 10000,
                 schedule: str = 'linear'):
        self.initial_p = initial_p
        self.final_p = final_p
        self.num_steps = num_steps
        self.schedule = schedule
        
        self.current_step = 0
        self.base_dropout = Dropout(p=initial_p)
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Update keep probability based on schedule
        p = self._get_scheduled_p()
        self.base_dropout.p = p
        self.base_dropout.training = self.training
        
        return self.base_dropout.forward(x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self.base_dropout.backward(grad_output)
    
    def step(self):
        """Call after each training step."""
        self.current_step = min(self.current_step + 1, self.num_steps)
    
    def _get_scheduled_p(self) -> float:
        """Compute current keep probability based on schedule."""
        progress = self.current_step / self.num_steps
        
        if self.schedule == 'linear':
            return self.initial_p + progress * (self.final_p - self.initial_p)
        elif self.schedule == 'cosine':
            return self.final_p + 0.5 * (self.initial_p - self.final_p) * (
                1 + np.cos(np.pi * progress))
        else:
            return self.initial_p


# =============================================================================
# MONTE CARLO DROPOUT
# =============================================================================

class MCDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Keeps dropout enabled at inference time and runs multiple forward
    passes to estimate:
    - Predictive mean (average prediction)
    - Predictive uncertainty (variance of predictions)
    
    Reference: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
    """
    
    def __init__(self, model, p: float = 0.5):
        """
        Args:
            model: A model that uses Dropout layers
            p: Dropout keep probability
        """
        self.model = model
        self.p = p
    
    def predict(self, x: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run multiple forward passes with dropout.
        
        Args:
            x: Input data
            n_samples: Number of forward passes
            
        Returns:
            mean: Averaged prediction
            variance: Uncertainty estimate
        """
        # Ensure model is in training mode (dropout active)
        self._set_training(True)
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model(x)
            predictions.append(pred)
        
        predictions = np.stack(predictions, axis=0)
        
        mean = predictions.mean(axis=0)
        variance = predictions.var(axis=0)
        
        return mean, variance
    
    def predict_with_entropy(self, x: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and epistemic uncertainty via entropy.
        
        For classification, entropy is a better uncertainty measure.
        """
        mean, variance = self.predict(x, n_samples)
        
        # For classification (softmax output)
        # Compute predictive entropy
        eps = 1e-10
        probs = np.clip(mean, eps, 1 - eps)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        
        return mean, entropy
    
    def _set_training(self, mode: bool):
        """Set all dropout layers to training mode."""
        if hasattr(self.model, 'dropout_layers'):
            for layer in self.model.dropout_layers:
                layer.training = mode


# =============================================================================
# NEURAL NETWORK WITH DROPOUT
# =============================================================================

class Linear:
    """Simple linear layer for building networks."""
    
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.input = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.W.T + self.b
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad_input = grad_output @ self.W
        grad_W = grad_output.T @ self.input
        grad_b = grad_output.sum(axis=0)
        return grad_input, grad_W, grad_b


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def relu_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """ReLU backward pass."""
    return grad_output * (x > 0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class DropoutNetwork:
    """
    Complete neural network with dropout for educational purposes.
    
    Architecture:
        Input → Dense → ReLU → Dropout → Dense → ReLU → Dropout → Output
        
    This demonstrates how to properly integrate dropout into a network.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 dropout_p: float = 0.5):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
            dropout_p: Dropout keep probability
        """
        self.layers = []
        self.dropout_layers = []
        self.activations = []
        
        # Build layers
        all_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(all_sizes) - 1):
            # Linear layer
            self.layers.append(Linear(all_sizes[i], all_sizes[i+1]))
            
            # Dropout (except for last layer)
            if i < len(all_sizes) - 2:
                dropout = Dropout(p=dropout_p)
                self.dropout_layers.append(dropout)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        self.activations = []
        self.pre_activations = []
        
        for i, layer in enumerate(self.layers):
            # Linear
            x = layer.forward(x)
            self.pre_activations.append(x.copy())
            
            # ReLU + Dropout (except last layer)
            if i < len(self.layers) - 1:
                x = relu(x)
                self.activations.append(x.copy())
                x = self.dropout_layers[i].forward(x)
        
        return softmax(x)
    
    def train(self):
        """Set to training mode."""
        for dropout in self.dropout_layers:
            dropout.train()
    
    def eval(self):
        """Set to evaluation mode."""
        for dropout in self.dropout_layers:
            dropout.eval()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_dropout_rates(X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          dropout_rates: List[float] = [0.0, 0.3, 0.5, 0.7]) -> dict:
    """
    Train networks with different dropout rates and compare.
    
    Returns:
        Dictionary with training and validation accuracies for each rate.
    """
    results = {}
    
    for p in dropout_rates:
        keep_prob = 1.0 if p == 0 else 1 - p
        
        print(f"\nTraining with dropout keep_prob = {keep_prob:.1f}...")
        
        # Create network
        net = DropoutNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=[256, 128],
            output_size=10,
            dropout_p=keep_prob
        )
        
        # Simple training loop (simplified)
        learning_rate = 0.01
        
        for epoch in range(10):
            net.train()
            
            # Forward
            logits = net.forward(X_train)
            
            # Compute accuracy
            train_acc = (logits.argmax(axis=1) == y_train).mean()
            
            net.eval()
            val_logits = net.forward(X_val)
            val_acc = (val_logits.argmax(axis=1) == y_val).mean()
        
        results[keep_prob] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc
        }
        
        print(f"  Train: {train_acc:.3f}, Val: {val_acc:.3f}, Gap: {train_acc - val_acc:.3f}")
    
    return results


def visualize_dropout_mask(p: float = 0.5, size: Tuple[int, int] = (8, 8)):
    """Generate and display a dropout mask."""
    mask = (np.random.rand(*size) < p).astype(np.float32)
    
    kept = mask.sum()
    total = mask.size
    
    print(f"Dropout mask (p={p}):")
    print(f"Kept: {int(kept)}/{total} ({100*kept/total:.1f}%)")
    
    # ASCII visualization (using ASCII-compatible characters)
    for row in mask:
        print(''.join(['#' if v else '.' for v in row]))
    
    return mask


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DROPOUT IMPLEMENTATION DEMO")
    print("=" * 60)
    
    # 1. Basic dropout demonstration
    print("\n1. Basic Dropout:")
    print("-" * 40)
    
    dropout = Dropout(p=0.5)
    x = np.ones((2, 8))
    
    print(f"Input: {x[0]}")
    
    # Training mode
    dropout.train()
    y_train = dropout.forward(x)
    print(f"Training output: {y_train[0]}")
    print(f"(Note: Values are either 0 or 2x due to inverted scaling)")
    
    # Eval mode
    dropout.eval()
    y_eval = dropout.forward(x)
    print(f"Eval output: {y_eval[0]}")
    print(f"(Note: No dropout in eval mode)")
    
    # 2. Dropout2D for CNNs
    print("\n2. Spatial Dropout (Dropout2D):")
    print("-" * 40)
    
    dropout2d = Dropout2D(p=0.5)
    x_cnn = np.ones((1, 4, 3, 3))  # batch=1, channels=4, h=3, w=3
    
    dropout2d.train()
    y_cnn = dropout2d.forward(x_cnn)
    
    print(f"Input channels: 4")
    for c in range(4):
        is_kept = y_cnn[0, c, 0, 0] > 0
        print(f"  Channel {c}: {'KEPT (scaled 2x)' if is_kept else 'DROPPED'}")
    
    # 3. Visualize dropout masks
    print("\n3. Dropout Mask Visualization:")
    print("-" * 40)
    
    for p in [0.8, 0.5, 0.2]:
        print(f"\nKeep probability p = {p}:")
        visualize_dropout_mask(p=p)
    
    # 4. Neural Network with Dropout
    print("\n4. DropoutNetwork Demo:")
    print("-" * 40)
    
    # Create network
    net = DropoutNetwork(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        dropout_p=0.5
    )
    
    # Random input
    x = np.random.randn(4, 784)
    
    # Training: Different outputs each time
    net.train()
    print("Training mode (dropout active):")
    for i in range(3):
        y = net(x)
        print(f"  Forward pass {i+1}: class probabilities sum = {y[0].sum():.3f}")
    
    # Eval: Same output each time
    net.eval()
    print("\nEval mode (dropout disabled):")
    for i in range(3):
        y = net(x)
        print(f"  Forward pass {i+1}: class probabilities = {y[0][:3]}...")
    
    # 5. MC Dropout for uncertainty
    print("\n5. MC Dropout (Uncertainty Estimation):")
    print("-" * 40)
    
    # Simple model for demo
    net.train()  # Keep dropout active
    
    x_test = np.random.randn(1, 784)
    
    # Multiple predictions
    predictions = []
    for _ in range(20):
        pred = net(x_test)
        predictions.append(pred)
    
    predictions = np.stack(predictions)
    
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    print(f"Mean prediction (top 3 classes): {mean_pred[0][:3]}")
    print(f"Std (uncertainty) per class: {std_pred[0][:3]}")
    print(f"Predicted class: {mean_pred[0].argmax()}")
    print(f"Uncertainty: {std_pred[0].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Explore the code to learn more.")
    print("=" * 60)
