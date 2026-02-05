"""
Exercise 4: Early Stopping Implementation
==========================================

Goal: Implement early stopping to prevent overfitting by stopping 
training when validation loss stops improving.

Your Task:
- Fill in the TODOs below to complete the early stopping implementation
- Test your implementation with the provided tests
- Compare with the reference solution

Learning Objectives:
1. Understand why early stopping prevents overfitting
2. Implement patience-based stopping criterion
3. Track best model state for restoration
4. Handle edge cases properly

Time: 25-35 minutes
Difficulty: Medium

Key Concept:
    Early stopping monitors validation loss during training.
    If validation loss doesn't improve for 'patience' epochs,
    we stop training and restore the best model weights.
    
    Timeline Example (patience=3):
        Epoch 1: val_loss=2.5  <- new best!
        Epoch 2: val_loss=2.3  <- new best!
        Epoch 3: val_loss=2.4  <- no improvement (wait 1)
        Epoch 4: val_loss=2.5  <- no improvement (wait 2)
        Epoch 5: val_loss=2.6  <- no improvement (wait 3)
        Epoch 6: STOP! Restore weights from Epoch 2
"""

import numpy as np
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Usage:
        early_stop = EarlyStopping(patience=5, min_delta=0.001)
        
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model)
            val_loss = validate(model)
            
            if early_stop(val_loss, model_weights):
                print(f"Early stopping at epoch {epoch}")
                model.load_weights(early_stop.get_best_weights())
                break
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print messages
            
        Example:
            patience=5 means: stop if no improvement for 5 epochs
            min_delta=0.001 means: improvement must be at least 0.001
        """
        # TODO 1: Store parameters
        # self.patience = patience
        # self.min_delta = min_delta
        # self.verbose = verbose
        
        # TODO 2: Initialize tracking variables
        # self.best_loss = float('inf')  # Best validation loss seen
        # self.wait = 0                   # Epochs since improvement
        # self.stopped_epoch = 0          # Epoch when stopped
        # self.best_weights = None        # Weights at best loss
        
        pass  # Remove this when you implement
    
    def __call__(self, val_loss: float, model_weights: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model_weights: Dictionary of model weights (optional, for restoration)
            
        Returns:
            True if training should stop, False otherwise
            
        Logic:
            1. Check if val_loss improved by at least min_delta
            2. If improved: 
               - Update best_loss
               - Reset wait counter
               - Save weights
            3. If not improved:
               - Increment wait counter
               - Check if wait >= patience
        """
        # TODO 3: Check if this is an improvement
        # An improvement means: val_loss < best_loss - min_delta
        # is_improvement = ???
        
        # TODO 4: If improved, update best and reset counter
        # if is_improvement:
        #     self.best_loss = val_loss
        #     self.wait = 0
        #     if model_weights is not None:
        #         # Deep copy the weights!
        #         self.best_weights = {k: v.copy() for k, v in model_weights.items()}
        #     if self.verbose:
        #         print(f"  Validation improved to {val_loss:.6f}")
        #     return False
        
        # TODO 5: If not improved, increment wait counter
        # self.wait += 1
        # if self.verbose:
        #     print(f"  No improvement. Patience: {self.wait}/{self.patience}")
        
        # TODO 6: Check if we should stop
        # if self.wait >= self.patience:
        #     return True
        # return False
        
        pass  # Remove this when you implement
    
    def get_best_weights(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the best weights, or None if not saved."""
        # TODO 7: Return best weights
        # return self.best_weights
        pass  # Remove this when you implement
    
    def get_best_loss(self) -> float:
        """Return the best validation loss seen."""
        # TODO 8: Return best loss
        # return self.best_loss
        pass  # Remove this when you implement
    
    def reset(self):
        """Reset the early stopping state."""
        # TODO 9: Reset all tracking variables
        # self.best_loss = float('inf')
        # self.wait = 0
        # self.stopped_epoch = 0
        # self.best_weights = None
        pass  # Remove this when you implement


def simulate_training_with_early_stopping(
    train_losses: list,
    val_losses: list, 
    patience: int = 3,
    min_delta: float = 0.0
) -> Dict[str, Any]:
    """
    Simulate training with early stopping.
    
    Args:
        train_losses: Pre-defined training losses for each epoch
        val_losses: Pre-defined validation losses for each epoch
        patience: Patience for early stopping
        min_delta: Minimum improvement required
        
    Returns:
        Dictionary with:
        - stopped_epoch: Epoch where training stopped
        - best_epoch: Epoch with best validation loss
        - best_val_loss: Best validation loss
        - epochs_trained: Number of epochs actually trained
    """
    # TODO 10: Initialize early stopping
    # early_stop = EarlyStopping(patience=patience, min_delta=min_delta, verbose=False)
    
    # TODO 11: Simulate training loop
    # for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
    #     # Create fake weights (in real training, these would be model.get_weights())
    #     fake_weights = {'W': np.array([epoch])}
    #     
    #     # Check if should stop
    #     if early_stop(val_loss, fake_weights):
    #         return {
    #             'stopped_epoch': epoch,
    #             'best_epoch': ???,  # Epoch with best weights
    #             'best_val_loss': early_stop.get_best_loss(),
    #             'epochs_trained': epoch + 1
    #         }
    #
    # # If we get here, training completed without early stopping
    # return {
    #     'stopped_epoch': len(val_losses) - 1,
    #     'best_epoch': ???,
    #     'best_val_loss': early_stop.get_best_loss(),
    #     'epochs_trained': len(val_losses)
    # }
    
    pass  # Remove this when you implement


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_early_stopping_basic():
    """Test basic early stopping functionality."""
    print("Testing basic early stopping...")
    
    # Test 1: Should stop after patience epochs of no improvement
    es = EarlyStopping(patience=3, verbose=False)
    
    # Epoch 0: improvement (2.0 < inf)
    should_stop = es(2.0)
    assert not should_stop, "Should not stop on first improvement"
    
    # Epoch 1: improvement (1.8 < 2.0)
    should_stop = es(1.8)
    assert not should_stop, "Should not stop on improvement"
    
    # Epoch 2: no improvement (2.0 > 1.8)
    should_stop = es(2.0)
    assert not should_stop, "Should not stop yet (patience=1)"
    
    # Epoch 3: no improvement (2.1 > 1.8)
    should_stop = es(2.1)
    assert not should_stop, "Should not stop yet (patience=2)"
    
    # Epoch 4: no improvement (2.2 > 1.8)
    should_stop = es(2.2)
    assert should_stop, "Should stop now (patience=3 reached)"
    
    print("  [ok] Test 1: Stops after patience epochs")
    
    # Test 2: Best loss tracking
    assert abs(es.get_best_loss() - 1.8) < 1e-6, f"Best loss should be 1.8, got {es.get_best_loss()}"
    print("  [ok] Test 2: Best loss tracked correctly")
    
    print("[ok] All basic tests passed!\n")


def test_min_delta():
    """Test min_delta parameter."""
    print("Testing min_delta...")
    
    es = EarlyStopping(patience=2, min_delta=0.1, verbose=False)
    
    # Start
    es(2.0)  # Best = 2.0
    
    # 1.95 is not 0.1 better than 2.0, so no improvement
    es(1.95)  # wait = 1
    assert es.wait == 1, f"Should wait (1.95 not 0.1 better than 2.0)"
    
    # 1.89 is 0.1+ better than 2.0, so improvement!
    es(1.89)  # New best, wait = 0
    assert es.wait == 0, "Should reset (1.89 is 0.1+ better than 2.0)"
    assert abs(es.get_best_loss() - 1.89) < 1e-6, "Best should update to 1.89"
    
    print("  [ok] min_delta correctly requires minimum improvement")
    print("[ok] min_delta tests passed!\n")


def test_weight_saving():
    """Test that weights are saved correctly."""
    print("Testing weight saving...")
    
    es = EarlyStopping(patience=3, verbose=False)
    
    # Epoch 0: best weights
    weights_0 = {'W': np.array([[1, 2], [3, 4]]), 'b': np.array([1, 2])}
    es(2.0, weights_0)
    
    # Epoch 1: new best weights
    weights_1 = {'W': np.array([[5, 6], [7, 8]]), 'b': np.array([3, 4])}
    es(1.5, weights_1)
    
    # Epoch 2: no improvement, weights not saved
    weights_2 = {'W': np.array([[9, 10], [11, 12]]), 'b': np.array([5, 6])}
    es(2.0, weights_2)
    
    # Check best weights are from epoch 1
    best = es.get_best_weights()
    assert np.allclose(best['W'], weights_1['W']), "Best weights should be from epoch 1"
    assert np.allclose(best['b'], weights_1['b']), "Best bias should be from epoch 1"
    
    # Modify original weights (should not affect saved copy)
    weights_1['W'][0, 0] = 999
    assert best['W'][0, 0] != 999, "Weights should be deep copied"
    
    print("  [ok] Test 1: Best weights saved correctly")
    print("  [ok] Test 2: Weights are deep copied")
    print("[ok] Weight saving tests passed!\n")


def test_reset():
    """Test reset functionality."""
    print("Testing reset...")
    
    es = EarlyStopping(patience=2, verbose=False)
    
    # Use it
    es(2.0, {'W': np.array([1])})
    es(1.5, {'W': np.array([2])})
    es(2.0)
    
    # Reset
    es.reset()
    
    # Check everything is reset
    assert es.best_loss == float('inf'), "best_loss should reset to inf"
    assert es.wait == 0, "wait should reset to 0"
    assert es.best_weights is None, "best_weights should reset to None"
    
    print("  [ok] Reset works correctly")
    print("[ok] Reset tests passed!\n")


def test_simulation():
    """Test training simulation."""
    print("Testing training simulation...")
    
    # Scenario: Best at epoch 2, no improvement after
    train_losses = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    val_losses =   [2.0, 1.5, 1.3, 1.5, 1.6, 1.7]  # Best at epoch 2 (1.3)
    
    result = simulate_training_with_early_stopping(
        train_losses, val_losses,
        patience=3, min_delta=0.0
    )
    
    # Should stop at epoch 5 (3 epochs without improvement after epoch 2)
    assert result['stopped_epoch'] == 5, f"Should stop at epoch 5, got {result['stopped_epoch']}"
    assert abs(result['best_val_loss'] - 1.3) < 1e-6, f"Best val loss should be 1.3"
    
    print(f"  [ok] Stopped at epoch {result['stopped_epoch']}")
    print(f"  [ok] Best val loss: {result['best_val_loss']}")
    print("[ok] Simulation tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 4: Early Stopping Implementation")
    print("="*60 + "\n")
    
    try:
        test_early_stopping_basic()
        test_min_delta()
        test_weight_saving()
        test_reset()
        test_simulation()
        print("All tests passed!")
    except AssertionError as e:
        print(f"FAIL - Test failed: {e}")
    except Exception as e:
        print(f"FAIL - Error: {e}")
        import traceback
        traceback.print_exc()
