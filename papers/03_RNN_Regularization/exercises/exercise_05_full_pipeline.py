"""
Exercise 5: Full Regularization Pipeline
==========================================

Goal: Combine ALL regularization techniques into one complete training pipeline.

Your Task:
- Fill in the TODOs below to build a complete regularized RNN
- Use: Dropout, Layer Normalization, Weight Decay, Early Stopping
- Test your implementation with the provided tests

Learning Objectives:
1. Integrate multiple regularization techniques
2. Understand how they work together
3. Build a production-ready training loop
4. See the combined effect on overfitting

Time: 40-50 minutes
Difficulty: Hard

The Four Regularization Techniques:
    1. DROPOUT: Randomly zero neurons (training only!)
    2. LAYER NORM: Normalize activations for stability
    3. WEIGHT DECAY: Penalize large weights (L2 regularization)
    4. EARLY STOPPING: Stop when validation loss stops improving
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class RegularizedRNNCell:
    """
    A single RNN cell with dropout and layer normalization.
    
    Architecture:
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        h_t = layer_norm(h_t)
        h_t = dropout(h_t)  # Only during training!
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        dropout_rate: float = 0.3,
        weight_decay: float = 0.0001
    ):
        """
        Initialize the regularized RNN cell.
        
        Args:
            input_size: Dimension of input vectors
            hidden_size: Dimension of hidden state
            dropout_rate: Probability of dropping a neuron (0.0 to 1.0)
            weight_decay: L2 regularization coefficient
        """
        # TODO 1: Store parameters
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.dropout_rate = dropout_rate
        # self.weight_decay = weight_decay
        
        # TODO 2: Initialize weights (Xavier initialization)
        # Xavier: scale by sqrt(2 / (fan_in + fan_out))
        # self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        # self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        # self.b_h = np.zeros(hidden_size)
        
        # TODO 3: Initialize layer norm parameters
        # self.gamma = np.ones(hidden_size)   # Scale
        # self.beta = np.zeros(hidden_size)   # Shift
        # self.epsilon = 1e-5                  # Numerical stability
        
        pass  # Remove this when you implement
    
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Formula:
            mean = mean(x)
            var = var(x)
            x_norm = (x - mean) / sqrt(var + epsilon)
            output = gamma * x_norm + beta
        """
        # TODO 4: Implement layer normalization
        # mean = np.mean(x, axis=-1, keepdims=True)
        # var = np.var(x, axis=-1, keepdims=True)
        # x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        # return self.gamma * x_norm + self.beta
        
        pass  # Remove this when you implement
    
    def dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout.
        
        During training: Randomly zero elements with probability dropout_rate
                        Scale remaining by 1/(1-dropout_rate)
        During inference: Return x unchanged
        """
        # TODO 5: Implement dropout with inverted scaling
        # if not training or self.dropout_rate == 0:
        #     return x
        # 
        # mask = np.random.rand(*x.shape) > self.dropout_rate
        # return (x * mask) / (1 - self.dropout_rate)
        
        pass  # Remove this when you implement
    
    def forward(
        self, 
        x: np.ndarray, 
        h_prev: np.ndarray, 
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass of regularized RNN cell.
        
        Args:
            x: Input at current timestep (batch_size, input_size)
            h_prev: Hidden state from previous timestep (batch_size, hidden_size)
            training: Whether we're training (affects dropout)
            
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        # TODO 6: Compute raw hidden state
        # h_raw = np.tanh(self.W_hh @ h_prev.T + self.W_xh @ x.T + self.b_h.reshape(-1, 1))
        # h_raw = h_raw.T  # Back to (batch_size, hidden_size)
        
        # TODO 7: Apply layer normalization
        # h_norm = self.layer_norm(h_raw)
        
        # TODO 8: Apply dropout (only during training!)
        # h = self.dropout(h_norm, training=training)
        
        # TODO 9: Return final hidden state
        # return h
        
        pass  # Remove this when you implement
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return all weights as a dictionary."""
        # TODO 10: Return weights dictionary
        # return {
        #     'W_xh': self.W_xh,
        #     'W_hh': self.W_hh,
        #     'b_h': self.b_h,
        #     'gamma': self.gamma,
        #     'beta': self.beta
        # }
        pass  # Remove this when you implement
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights from a dictionary."""
        # TODO 11: Set weights from dictionary
        # self.W_xh = weights['W_xh'].copy()
        # self.W_hh = weights['W_hh'].copy()
        # self.b_h = weights['b_h'].copy()
        # self.gamma = weights['gamma'].copy()
        # self.beta = weights['beta'].copy()
        pass  # Remove this when you implement
    
    def compute_l2_penalty(self) -> float:
        """Compute L2 regularization penalty."""
        # TODO 12: Compute L2 penalty for W_xh and W_hh
        # penalty = 0.0
        # penalty += np.sum(self.W_xh ** 2)
        # penalty += np.sum(self.W_hh ** 2)
        # return (self.weight_decay / 2) * penalty
        pass  # Remove this when you implement


class EarlyStopping:
    """Early stopping handler (simplified version)."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, weights: Optional[Dict] = None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if weights is not None:
                self.best_weights = {k: v.copy() for k, v in weights.items()}
            return False
        
        self.wait += 1
        return self.wait >= self.patience
    
    def get_best_weights(self) -> Optional[Dict]:
        return self.best_weights


def train_regularized_rnn(
    rnn_cell: RegularizedRNNCell,
    train_sequences: List[np.ndarray],
    train_targets: List[np.ndarray],
    val_sequences: List[np.ndarray],
    val_targets: List[np.ndarray],
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    patience: int = 5
) -> Dict:
    """
    Train the regularized RNN with early stopping.
    
    This is a simplified training loop that demonstrates
    how all regularization techniques work together.
    
    Args:
        rnn_cell: The RegularizedRNNCell to train
        train_sequences: List of input sequences for training
        train_targets: List of target values for training  
        val_sequences: List of input sequences for validation
        val_targets: List of target values for validation
        learning_rate: Learning rate for gradient descent
        max_epochs: Maximum number of training epochs
        patience: Patience for early stopping
        
    Returns:
        Dictionary with training history
    """
    # TODO 13: Initialize early stopping
    # early_stop = EarlyStopping(patience=patience)
    
    # TODO 14: Initialize history tracking
    # history = {
    #     'train_loss': [],
    #     'val_loss': [],
    #     'epochs_trained': 0,
    #     'stopped_early': False
    # }
    
    # TODO 15: Training loop
    # for epoch in range(max_epochs):
    #     # Training phase (with dropout)
    #     train_loss = 0.0
    #     for seq, target in zip(train_sequences, train_targets):
    #         h = np.zeros((1, rnn_cell.hidden_size))
    #         for t in range(len(seq)):
    #             x = seq[t:t+1]  # (1, input_size)
    #             h = rnn_cell.forward(x, h, training=True)
    #         
    #         # Simple MSE loss
    #         loss = np.mean((h - target) ** 2)
    #         
    #         # Add L2 penalty
    #         loss += rnn_cell.compute_l2_penalty()
    #         train_loss += loss
    #     
    #     train_loss /= len(train_sequences)
    #     
    #     # Validation phase (no dropout!)
    #     val_loss = 0.0
    #     for seq, target in zip(val_sequences, val_targets):
    #         h = np.zeros((1, rnn_cell.hidden_size))
    #         for t in range(len(seq)):
    #             x = seq[t:t+1]
    #             h = rnn_cell.forward(x, h, training=False)  # No dropout!
    #         
    #         loss = np.mean((h - target) ** 2)
    #         val_loss += loss
    #     
    #     val_loss /= len(val_sequences)
    #     
    #     # Record history
    #     history['train_loss'].append(train_loss)
    #     history['val_loss'].append(val_loss)
    #     
    #     # Check early stopping
    #     if early_stop(val_loss, rnn_cell.get_weights()):
    #         history['stopped_early'] = True
    #         history['epochs_trained'] = epoch + 1
    #         
    #         # Restore best weights
    #         best_weights = early_stop.get_best_weights()
    #         if best_weights is not None:
    #             rnn_cell.set_weights(best_weights)
    #         break
    # else:
    #     history['epochs_trained'] = max_epochs
    # 
    # return history
    
    pass  # Remove this when you implement


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_rnn_cell_initialization():
    """Test RNN cell initialization."""
    print("Testing RNN cell initialization...")
    
    cell = RegularizedRNNCell(input_size=10, hidden_size=20, dropout_rate=0.3)
    
    # Check shapes
    assert cell.W_xh.shape == (20, 10), f"W_xh shape wrong: {cell.W_xh.shape}"
    assert cell.W_hh.shape == (20, 20), f"W_hh shape wrong: {cell.W_hh.shape}"
    assert cell.b_h.shape == (20,), f"b_h shape wrong: {cell.b_h.shape}"
    assert cell.gamma.shape == (20,), f"gamma shape wrong: {cell.gamma.shape}"
    assert cell.beta.shape == (20,), f"beta shape wrong: {cell.beta.shape}"
    
    print("  [ok] All shapes correct")
    print("[ok] Initialization tests passed!\n")


def test_layer_norm():
    """Test layer normalization."""
    print("Testing layer normalization...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10)
    
    # Test normalization
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    x_norm = cell.layer_norm(x)
    
    # After normalization, mean should be ~0 and std ~1
    # (with gamma=1 and beta=0)
    mean = np.mean(x_norm)
    std = np.std(x_norm)
    
    assert abs(mean) < 0.01, f"Mean should be ~0, got {mean}"
    assert abs(std - 1.0) < 0.01, f"Std should be ~1, got {std}"
    
    print("  [ok] Normalization works correctly")
    print("[ok] Layer norm tests passed!\n")


def test_dropout():
    """Test dropout."""
    print("Testing dropout...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=100, dropout_rate=0.5)
    
    np.random.seed(42)
    x = np.ones((1, 100))
    
    # Test training mode
    x_dropped = cell.dropout(x, training=True)
    zeros = np.sum(x_dropped == 0)
    assert zeros > 20, f"Should have some zeros, got {zeros}"
    
    # Test inference mode
    x_kept = cell.dropout(x, training=False)
    assert np.allclose(x_kept, x), "Inference mode should not drop"
    
    print("  [ok] Training mode drops neurons")
    print("  [ok] Inference mode keeps all neurons")
    print("[ok] Dropout tests passed!\n")


def test_forward_pass():
    """Test forward pass."""
    print("Testing forward pass...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10, dropout_rate=0.0)
    
    batch_size = 3
    x = np.random.randn(batch_size, 5)
    h_prev = np.zeros((batch_size, 10))
    
    h = cell.forward(x, h_prev, training=False)
    
    assert h.shape == (batch_size, 10), f"Output shape wrong: {h.shape}"
    
    print("  [ok] Forward pass works")
    print("[ok] Forward pass tests passed!\n")


def test_l2_penalty():
    """Test L2 penalty computation."""
    print("Testing L2 penalty...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10, weight_decay=0.1)
    
    # Set simple weights
    cell.W_xh = np.ones((10, 5))   # Sum of squares = 50
    cell.W_hh = np.ones((10, 10))  # Sum of squares = 100
    
    penalty = cell.compute_l2_penalty()
    expected = 0.1 / 2 * (50 + 100)  # = 7.5
    
    assert abs(penalty - expected) < 1e-6, f"Expected {expected}, got {penalty}"
    
    print("  [ok] L2 penalty computed correctly")
    print("[ok] L2 penalty tests passed!\n")


def test_weight_save_restore():
    """Test weight save and restore."""
    print("Testing weight save/restore...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10)
    
    # Save weights
    weights = cell.get_weights()
    original_W_xh = weights['W_xh'].copy()
    
    # Modify weights
    cell.W_xh = np.zeros((10, 5))
    
    # Restore weights
    cell.set_weights(weights)
    
    assert np.allclose(cell.W_xh, original_W_xh), "Weights not restored correctly"
    
    print("  [ok] Weights saved and restored correctly")
    print("[ok] Weight save/restore tests passed!\n")


def test_full_pipeline():
    """Test complete training pipeline."""
    print("Testing full pipeline...")
    
    # Create simple synthetic data
    np.random.seed(42)
    
    # Training data
    train_sequences = [np.random.randn(5, 3) for _ in range(10)]
    train_targets = [np.random.randn(1, 8) for _ in range(10)]
    
    # Validation data
    val_sequences = [np.random.randn(5, 3) for _ in range(5)]
    val_targets = [np.random.randn(1, 8) for _ in range(5)]
    
    # Create cell
    cell = RegularizedRNNCell(
        input_size=3, 
        hidden_size=8, 
        dropout_rate=0.2,
        weight_decay=0.001
    )
    
    # Train
    history = train_regularized_rnn(
        cell,
        train_sequences, train_targets,
        val_sequences, val_targets,
        max_epochs=20,
        patience=5
    )
    
    assert 'train_loss' in history, "Missing train_loss in history"
    assert 'val_loss' in history, "Missing val_loss in history"
    assert len(history['train_loss']) > 0, "No training history recorded"
    
    print(f"  [ok] Trained for {history['epochs_trained']} epochs")
    print(f"  [ok] Early stopped: {history['stopped_early']}")
    print("[ok] Full pipeline tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 5: Full Regularization Pipeline")
    print("="*60 + "\n")
    
    try:
        test_rnn_cell_initialization()
        test_layer_norm()
        test_dropout()
        test_forward_pass()
        test_l2_penalty()
        test_weight_save_restore()
        test_full_pipeline()
        print("All tests passed!")
    except AssertionError as e:
        print(f"FAIL - Test failed: {e}")
    except Exception as e:
        print(f"FAIL - Error: {e}")
        import traceback
        traceback.print_exc()
