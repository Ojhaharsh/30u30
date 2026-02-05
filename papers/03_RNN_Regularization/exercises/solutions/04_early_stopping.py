"""
Solution 4: Early Stopping Implementation
==========================================

Complete solution with explanations for each step.
"""

import numpy as np
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    SOLUTION 1: Complete Early Stopping Implementation
    
    Early stopping prevents overfitting by:
    1. Monitoring validation loss during training
    2. Saving the best model weights
    3. Stopping when no improvement for 'patience' epochs
    4. Restoring the best weights at the end
    
    Why it works:
    - Training loss always decreases (model memorizes training data)
    - Validation loss decreases, then increases (overfitting!)
    - We want to stop at the minimum of validation loss
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        SOLUTION 1a: Initialize early stopping
        
        Args:
            patience: How many epochs to wait for improvement
            min_delta: Minimum improvement required (prevents stopping on noise)
            verbose: Whether to print progress messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Tracking variables
        self.best_loss = float('inf')  # Best validation loss seen
        self.wait = 0                   # Epochs since last improvement
        self.stopped_epoch = 0          # When we stopped
        self.best_weights = None        # Weights at best loss
    
    def __call__(self, val_loss: float, model_weights: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        SOLUTION 1b: Check if training should stop
        
        Logic:
        1. Is val_loss better than best by at least min_delta?
        2. If yes: update best, reset counter, save weights
        3. If no: increment counter, check if >= patience
        """
        # Check if this is an improvement
        is_improvement = val_loss < self.best_loss - self.min_delta
        
        if is_improvement:
            # New best! Update and reset counter
            self.best_loss = val_loss
            self.wait = 0
            
            # Save weights (deep copy!)
            if model_weights is not None:
                self.best_weights = {k: v.copy() for k, v in model_weights.items()}
            
            if self.verbose:
                print(f"  Validation improved to {val_loss:.6f}")
            
            return False  # Don't stop
        
        # No improvement
        self.wait += 1
        
        if self.verbose:
            print(f"  No improvement. Patience: {self.wait}/{self.patience}")
        
        # Check if we should stop
        if self.wait >= self.patience:
            return True  # Stop!
        
        return False  # Keep going
    
    def get_best_weights(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the best weights."""
        return self.best_weights
    
    def get_best_loss(self) -> float:
        """Return the best validation loss."""
        return self.best_loss
    
    def reset(self):
        """Reset early stopping state."""
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None


def simulate_training_with_early_stopping(
    train_losses: list,
    val_losses: list,
    patience: int = 3,
    min_delta: float = 0.0
) -> Dict[str, Any]:
    """
    SOLUTION 2: Simulate training with early stopping
    
    This simulates a training loop to demonstrate how
    early stopping works in practice.
    """
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta, verbose=False)
    
    best_epoch = 0
    
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        # Create fake weights
        fake_weights = {'W': np.array([epoch])}
        
        # Track best epoch (for returning)
        if val_loss < early_stop.get_best_loss():
            best_epoch = epoch
        
        # Check if should stop
        if early_stop(val_loss, fake_weights):
            return {
                'stopped_epoch': epoch,
                'best_epoch': best_epoch,
                'best_val_loss': early_stop.get_best_loss(),
                'epochs_trained': epoch + 1
            }
    
    # Training completed without early stopping
    return {
        'stopped_epoch': len(val_losses) - 1,
        'best_epoch': best_epoch,
        'best_val_loss': early_stop.get_best_loss(),
        'epochs_trained': len(val_losses)
    }


def visualize_early_stopping():
    """
    SOLUTION 3: Visualize how early stopping works
    
    This shows a typical training curve and where
    early stopping would trigger.
    """
    print("Visualizing early stopping behavior...\n")
    
    # Simulate training/validation curves
    epochs = list(range(20))
    
    # Training loss always decreases (model fits training data)
    train_losses = [2.0 * np.exp(-0.15 * e) for e in epochs]
    
    # Validation loss decreases, then increases (overfitting!)
    val_losses = [
        2.0 * np.exp(-0.2 * e) + 0.03 * max(0, e - 7) ** 2
        for e in epochs
    ]
    
    print("Epoch | Train Loss | Val Loss  | Action")
    print("-" * 50)
    
    early_stop = EarlyStopping(patience=3, verbose=False)
    stopped_at = None
    
    for epoch in epochs:
        weights = {'W': np.array([epoch])}
        should_stop = early_stop(val_losses[epoch], weights)
        
        action = ""
        if val_losses[epoch] <= early_stop.get_best_loss():
            action = "[ok] New best!"
        elif stopped_at is None:
            action = f"Wait {early_stop.wait}/{early_stop.patience}"
        
        if should_stop and stopped_at is None:
            stopped_at = epoch
            action = "⚠ STOP!"
        
        print(f"  {epoch:2d}  |   {train_losses[epoch]:.4f}   |  {val_losses[epoch]:.4f}  | {action}")
    
    print(f"\nBest validation loss: {early_stop.get_best_loss():.4f}")
    print(f"Stopped at epoch: {stopped_at}")
    print("Without early stopping, validation would continue to get worse!")


def test_early_stopping_basic():
    """Test basic early stopping."""
    print("\nTesting basic early stopping...")
    
    es = EarlyStopping(patience=3, verbose=False)
    
    # Improvements
    assert not es(2.0), "Should not stop on first call"
    assert not es(1.8), "Should not stop on improvement"
    assert es.get_best_loss() == 1.8
    print("  [ok] Improvements tracked correctly")
    
    # No improvements
    assert not es(2.0), "Should not stop yet (wait=1)"
    assert not es(2.1), "Should not stop yet (wait=2)"
    assert es(2.2), "Should stop now (wait=3)"
    print("  [ok] Stops after patience epochs")
    
    print("[ok] Basic tests passed!")


def test_min_delta():
    """Test min_delta parameter."""
    print("\nTesting min_delta...")
    
    es = EarlyStopping(patience=2, min_delta=0.1, verbose=False)
    
    es(2.0)  # Best = 2.0
    es(1.95)  # Not 0.1 better, wait = 1
    assert es.wait == 1
    print("  [ok] Small improvement ignored")
    
    es(1.89)  # 0.11 better, reset!
    assert es.wait == 0
    print("  [ok] Sufficient improvement resets wait")
    
    print("[ok] min_delta tests passed!")


def test_weight_saving():
    """Test weight saving functionality."""
    print("\nTesting weight saving...")
    
    es = EarlyStopping(patience=3, verbose=False)
    
    # First best
    weights_0 = {'W': np.array([[1, 2], [3, 4]]), 'b': np.array([1, 2])}
    es(2.0, weights_0)
    
    # New best
    weights_1 = {'W': np.array([[5, 6], [7, 8]]), 'b': np.array([3, 4])}
    es(1.5, weights_1)
    
    # No improvement
    weights_2 = {'W': np.array([[9, 10], [11, 12]]), 'b': np.array([5, 6])}
    es(2.0, weights_2)
    
    # Check best weights
    best = es.get_best_weights()
    assert np.allclose(best['W'], weights_1['W'])
    print("  [ok] Best weights saved correctly")
    
    # Check deep copy
    weights_1['W'][0, 0] = 999
    assert best['W'][0, 0] != 999
    print("  [ok] Weights are deep copied")
    
    print("[ok] Weight saving tests passed!")


def test_simulation():
    """Test training simulation."""
    print("\nTesting simulation...")
    
    train_losses = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    val_losses = [2.0, 1.5, 1.3, 1.5, 1.6, 1.7]  # Best at epoch 2
    
    result = simulate_training_with_early_stopping(
        train_losses, val_losses,
        patience=3, min_delta=0.0
    )
    
    assert result['stopped_epoch'] == 5
    assert abs(result['best_val_loss'] - 1.3) < 1e-6
    print(f"  [ok] Stopped at epoch {result['stopped_epoch']}")
    print(f"  [ok] Best val loss: {result['best_val_loss']}")
    
    print("[ok] Simulation tests passed!")


if __name__ == "__main__":
    print("="*60)
    print("Solution 4: Early Stopping Implementation")
    print("="*60)
    
    visualize_early_stopping()
    test_early_stopping_basic()
    test_min_delta()
    test_weight_saving()
    test_simulation()
    
    print("\nAll tests passed!")
