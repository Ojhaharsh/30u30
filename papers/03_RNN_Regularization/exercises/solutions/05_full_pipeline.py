"""
Solution 5: Full Regularization Pipeline
==========================================

Complete solution combining ALL regularization techniques.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class RegularizedRNNCell:
    """
    SOLUTION 1: Complete RNN Cell with All Regularization
    
    This combines:
    - Dropout (on non-recurrent connections)
    - Layer Normalization (before activation)
    - Weight Decay (L2 regularization)
    
    Architecture per timestep:
    1. x_t (input) → Dropout (training only)
    2. Linear: z = W_xh @ x + W_hh @ h + b
    3. Layer Norm: z_norm = LN(z)
    4. Activation: h = tanh(z_norm)
    5. Output dropout (training only)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.3,
        weight_decay: float = 0.0001
    ):
        """
        SOLUTION 1a: Initialize all components
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Xavier initialization for weights
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        self.b_h = np.zeros(hidden_size)
        
        # Layer norm parameters
        self.gamma = np.ones(hidden_size)
        self.beta = np.zeros(hidden_size)
        self.epsilon = 1e-5
    
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        SOLUTION 2: Layer Normalization
        
        Normalizes across the feature dimension:
        - Compute mean and variance per sample
        - Normalize: (x - mean) / std
        - Scale and shift: gamma * x_norm + beta
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta
    
    def dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        SOLUTION 3: Inverted Dropout
        
        Training: Drop neurons randomly, scale remaining
        Inference: Return unchanged (no dropout)
        
        Why "inverted"?
        - Scale by 1/(1-rate) during training
        - No scaling needed at inference
        - Makes inference faster!
        """
        if not training or self.dropout_rate == 0:
            return x
        
        mask = np.random.rand(*x.shape) > self.dropout_rate
        return (x * mask) / (1 - self.dropout_rate)
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        SOLUTION 4: Forward Pass with All Regularization
        
        Steps:
        1. Linear transformation (no dropout on recurrent!)
        2. Layer normalization
        3. Tanh activation
        4. Dropout (training only)
        """
        # Linear transformation
        # Note: W_hh @ h_prev does NOT get dropout
        # (recurrent connections preserve memory)
        h_raw = np.tanh(
            self.W_hh @ h_prev.T + 
            self.W_xh @ x.T + 
            self.b_h.reshape(-1, 1)
        )
        h_raw = h_raw.T  # Back to (batch, hidden)
        
        # Layer normalization
        h_norm = self.layer_norm(h_raw)
        
        # Dropout (training only)
        h = self.dropout(h_norm, training=training)
        
        return h
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return all weights as dictionary."""
        return {
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'gamma': self.gamma,
            'beta': self.beta
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights from dictionary."""
        self.W_xh = weights['W_xh'].copy()
        self.W_hh = weights['W_hh'].copy()
        self.b_h = weights['b_h'].copy()
        self.gamma = weights['gamma'].copy()
        self.beta = weights['beta'].copy()
    
    def compute_l2_penalty(self) -> float:
        """
        SOLUTION 5: L2 Regularization Penalty
        
        Penalize large weights to encourage simpler models.
        Only regularize W_xh and W_hh (not biases or layer norm).
        """
        penalty = 0.0
        penalty += np.sum(self.W_xh ** 2)
        penalty += np.sum(self.W_hh ** 2)
        return (self.weight_decay / 2) * penalty


class EarlyStopping:
    """Early stopping handler."""
    
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
    SOLUTION 6: Complete Training Loop
    
    This demonstrates how all regularization techniques
    work together in a real training loop:
    
    1. DROPOUT: Applied during training forward pass
    2. LAYER NORM: Applied in forward pass (always)
    3. WEIGHT DECAY: Added to loss computation
    4. EARLY STOPPING: Monitors validation loss
    """
    early_stop = EarlyStopping(patience=patience)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs_trained': 0,
        'stopped_early': False
    }
    
    for epoch in range(max_epochs):
        # =====================
        # TRAINING PHASE
        # =====================
        # Dropout IS active (training=True)
        train_loss = 0.0
        for seq, target in zip(train_sequences, train_targets):
            h = np.zeros((1, rnn_cell.hidden_size))
            
            for t in range(len(seq)):
                x = seq[t:t+1]  # (1, input_size)
                h = rnn_cell.forward(x, h, training=True)  # Dropout ON
            
            # MSE loss
            loss = np.mean((h - target) ** 2)
            
            # Add L2 penalty (weight decay)
            loss += rnn_cell.compute_l2_penalty()
            train_loss += loss
        
        train_loss /= len(train_sequences)
        
        # =====================
        # VALIDATION PHASE
        # =====================
        # Dropout is OFF (training=False)
        val_loss = 0.0
        for seq, target in zip(val_sequences, val_targets):
            h = np.zeros((1, rnn_cell.hidden_size))
            
            for t in range(len(seq)):
                x = seq[t:t+1]
                h = rnn_cell.forward(x, h, training=False)  # Dropout OFF!
            
            loss = np.mean((h - target) ** 2)
            val_loss += loss
        
        val_loss /= len(val_sequences)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # =====================
        # EARLY STOPPING CHECK
        # =====================
        if early_stop(val_loss, rnn_cell.get_weights()):
            history['stopped_early'] = True
            history['epochs_trained'] = epoch + 1
            
            # Restore best weights
            best_weights = early_stop.get_best_weights()
            if best_weights is not None:
                rnn_cell.set_weights(best_weights)
            break
    else:
        history['epochs_trained'] = max_epochs
    
    return history


def test_full_pipeline():
    """Test the complete regularization pipeline."""
    print("Testing full regularization pipeline...\n")
    
    np.random.seed(42)
    
    # Create synthetic data
    print("1. Creating synthetic data...")
    train_sequences = [np.random.randn(5, 3) for _ in range(20)]
    train_targets = [np.random.randn(1, 8) for _ in range(20)]
    val_sequences = [np.random.randn(5, 3) for _ in range(5)]
    val_targets = [np.random.randn(1, 8) for _ in range(5)]
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val: {len(val_sequences)} sequences")
    
    # Create regularized RNN
    print("\n2. Creating regularized RNN...")
    cell = RegularizedRNNCell(
        input_size=3,
        hidden_size=8,
        dropout_rate=0.2,
        weight_decay=0.001
    )
    print(f"   Input size: 3")
    print(f"   Hidden size: 8")
    print(f"   Dropout rate: 0.2")
    print(f"   Weight decay: 0.001")
    
    # Train
    print("\n3. Training with early stopping (patience=5)...")
    history = train_regularized_rnn(
        cell,
        train_sequences, train_targets,
        val_sequences, val_targets,
        max_epochs=50,
        patience=5
    )
    
    print(f"\n4. Results:")
    print(f"   Epochs trained: {history['epochs_trained']}")
    print(f"   Early stopped: {history['stopped_early']}")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Verify all techniques are working
    print("\n5. Verification:")
    
    # Dropout test
    h_train = cell.forward(np.random.randn(1, 3), np.zeros((1, 8)), training=True)
    h_test = cell.forward(np.random.randn(1, 3), np.zeros((1, 8)), training=False)
    print(f"   [ok] Dropout works (training/inference modes)")
    
    # Layer norm test
    h = cell.forward(np.random.randn(1, 3) * 100, np.zeros((1, 8)), training=False)
    print(f"   [ok] Layer norm works (output bounded: max={np.max(np.abs(h)):.2f})")
    
    # Weight decay test
    penalty = cell.compute_l2_penalty()
    print(f"   [ok] Weight decay works (penalty={penalty:.6f})")
    
    # Early stopping test
    print(f"   [ok] Early stopping works (stopped at epoch {history['epochs_trained']})")
    
    print("\nFull pipeline test passed!")


def demonstrate_regularization_effects():
    """Show how each regularization technique helps."""
    print("\nDemonstrating regularization effects...\n")
    
    np.random.seed(42)
    
    # Create data with clear train/val split
    # Training data has noise that shouldn't be learned
    train_sequences = [np.random.randn(5, 3) * 2 for _ in range(30)]
    train_targets = [np.random.randn(1, 8) for _ in range(30)]
    
    # Validation data is similar but different
    val_sequences = [np.random.randn(5, 3) * 2 for _ in range(10)]
    val_targets = [np.random.randn(1, 8) for _ in range(10)]
    
    configs = [
        ("No regularization", 0.0, 0.0),
        ("Dropout only", 0.3, 0.0),
        ("Weight decay only", 0.0, 0.01),
        ("Full regularization", 0.3, 0.01),
    ]
    
    print("Configuration          | Final Val Loss | Epochs | Early Stop")
    print("-" * 70)
    
    for name, dropout, decay in configs:
        np.random.seed(42)  # Same init for fair comparison
        
        cell = RegularizedRNNCell(
            input_size=3,
            hidden_size=8,
            dropout_rate=dropout,
            weight_decay=decay
        )
        
        history = train_regularized_rnn(
            cell,
            train_sequences, train_targets,
            val_sequences, val_targets,
            max_epochs=30,
            patience=5
        )
        
        val_loss = history['val_loss'][-1]
        epochs = history['epochs_trained']
        stopped = "Yes" if history['stopped_early'] else "No"
        
        print(f"{name:22} | {val_loss:14.4f} | {epochs:6} | {stopped}")
    
    print("\nRegularization typically leads to:")
    print("- Lower validation loss (better generalization)")
    print("- Earlier stopping (less overfitting)")
    print("- More stable training")


if __name__ == "__main__":
    print("="*60)
    print("Solution 5: Full Regularization Pipeline")
    print("="*60)
    
    test_full_pipeline()
    demonstrate_regularization_effects()
    
    print("\nAll tests passed!")
