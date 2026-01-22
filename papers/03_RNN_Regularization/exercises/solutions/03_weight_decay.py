"""
Solution 3: Weight Decay (L2 Regularization)
=============================================

Complete solution with explanations for each step.
"""

import numpy as np


def compute_l2_penalty(weights_list: list, weight_decay: float) -> float:
    """
    SOLUTION 1: Compute L2 regularization penalty
    
    Formula: penalty = (λ/2) * Σ(w²)
    
    Why the 1/2?
    - Makes the gradient cleaner
    - d/dw[(λ/2)*w²] = λ*w (no extra factor of 2)
    
    Args:
        weights_list: List of weight matrices
        weight_decay: Regularization coefficient (λ)
        
    Returns:
        penalty: Scalar L2 penalty
    """
    # SOLUTION 1a: Handle empty list
    if len(weights_list) == 0 or weight_decay == 0:
        return 0.0
    
    # SOLUTION 1b: Sum squared weights across all matrices
    penalty = 0.0
    for W in weights_list:
        penalty += np.sum(W ** 2)
    
    # SOLUTION 1c: Apply weight decay coefficient with 1/2
    penalty = (weight_decay / 2) * penalty
    
    return penalty


def compute_l2_gradient(W: np.ndarray, weight_decay: float) -> np.ndarray:
    """
    SOLUTION 2: Compute gradient contribution from L2 regularization
    
    The penalty is: (λ/2) * Σ(w²)
    The gradient w.r.t. w is: λ * w
    
    This gradient gets ADDED to the gradient from the loss function.
    
    Effect: Pulls weights toward zero proportionally to their magnitude.
    Large weights get pulled more strongly than small weights.
    """
    return weight_decay * W


def regularized_loss(model_loss: float, weights_list: list, weight_decay: float) -> float:
    """
    SOLUTION 3: Compute total loss with L2 regularization
    
    Total loss = Model loss + L2 penalty
    
    Args:
        model_loss: Loss from the model (cross-entropy, MSE, etc.)
        weights_list: List of weight matrices
        weight_decay: Regularization coefficient
        
    Returns:
        total_loss: Combined loss
    """
    # SOLUTION 3: Simply add penalty to model loss
    penalty = compute_l2_penalty(weights_list, weight_decay)
    return model_loss + penalty


def apply_weight_decay_to_gradient(dW: np.ndarray, W: np.ndarray, weight_decay: float) -> np.ndarray:
    """
    SOLUTION 4: Add weight decay contribution to gradient
    
    This is what happens in the optimizer:
    
    1. Compute gradient from loss: dW = d(loss)/dW
    2. Add regularization: dW += λ * W
    3. Update: W = W - lr * dW
    
    Expanding step 3:
        W = W - lr * (dW + λ * W)
        W = W - lr*dW - lr*λ*W
        W = W*(1 - lr*λ) - lr*dW
        
    The (1 - lr*λ) factor means weights DECAY each step!
    That's why it's called "weight decay".
    """
    return dW + weight_decay * W


class L2Regularizer:
    """
    SOLUTION 5: Complete L2 Regularizer class
    
    Convenient wrapper for applying L2 regularization
    consistently across a model.
    """
    
    def __init__(self, weight_decay: float = 0.0001):
        """
        Args:
            weight_decay: Regularization strength
            
        Common values:
        - 0.01: Strong regularization
        - 0.001: Moderate regularization
        - 0.0001: Light regularization (typical default)
        - 0: No regularization
        """
        self.weight_decay = weight_decay
    
    def penalty(self, weights_list: list) -> float:
        """Compute L2 penalty."""
        return compute_l2_penalty(weights_list, self.weight_decay)
    
    def augment_gradient(self, dW: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Add L2 gradient to existing gradient."""
        return apply_weight_decay_to_gradient(dW, W, self.weight_decay)
    
    def total_loss(self, model_loss: float, weights_list: list) -> float:
        """Compute total loss including penalty."""
        return regularized_loss(model_loss, weights_list, self.weight_decay)


def demonstrate_weight_decay_effect():
    """
    SOLUTION 6: Show how weight decay affects training
    
    This demonstrates why weight decay prevents overfitting:
    - Without decay: Weights can grow unboundedly
    - With decay: Weights are pulled toward zero
    - Smaller weights = simpler model = better generalization
    """
    print("Demonstrating weight decay effect...")
    
    np.random.seed(42)
    
    # Initial weights
    W = np.random.randn(5, 5)
    dW = np.random.randn(5, 5) * 0.01  # Small gradient
    learning_rate = 0.1
    
    print(f"\nInitial weight magnitude: {np.linalg.norm(W):.4f}")
    
    # Without weight decay
    W_no_decay = W.copy()
    for _ in range(100):
        W_no_decay = W_no_decay - learning_rate * dW
    print(f"After 100 steps (no decay): {np.linalg.norm(W_no_decay):.4f}")
    
    # With weight decay
    W_decay = W.copy()
    weight_decay = 0.01
    for _ in range(100):
        dW_total = dW + weight_decay * W_decay  # Add decay gradient
        W_decay = W_decay - learning_rate * dW_total
    print(f"After 100 steps (decay=0.01): {np.linalg.norm(W_decay):.4f}")
    
    # Stronger weight decay
    W_strong = W.copy()
    weight_decay = 0.1
    for _ in range(100):
        dW_total = dW + weight_decay * W_strong
        W_strong = W_strong - learning_rate * dW_total
    print(f"After 100 steps (decay=0.1): {np.linalg.norm(W_strong):.4f}")
    
    print("\n✓ Weight decay keeps weights smaller!")


def test_compute_l2_penalty():
    """Test L2 penalty computation."""
    print("\nTesting compute_l2_penalty...")
    
    # Test 1: Simple case
    W1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # Sum of squares = 30
    penalty = compute_l2_penalty([W1], weight_decay=0.1)
    expected = 0.1 / 2 * 30  # = 1.5
    assert abs(penalty - expected) < 1e-6
    print(f"  ✓ Simple case: {penalty}")
    
    # Test 2: Multiple matrices
    W1 = np.ones((2, 2))  # Sum = 4
    W2 = np.ones((3, 3))  # Sum = 9
    penalty = compute_l2_penalty([W1, W2], weight_decay=0.01)
    expected = 0.01 / 2 * 13
    assert abs(penalty - expected) < 1e-6
    print(f"  ✓ Multiple matrices: {penalty}")
    
    # Test 3: Zero decay
    penalty = compute_l2_penalty([np.random.randn(10, 10)], weight_decay=0.0)
    assert penalty == 0.0
    print("  ✓ Zero weight decay")
    
    print("✓ All penalty tests passed!")


def test_gradients():
    """Test gradient computation."""
    print("\nTesting gradient computation...")
    
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight_decay = 0.1
    
    # Test gradient
    dW_reg = compute_l2_gradient(W, weight_decay)
    expected = 0.1 * W
    assert np.allclose(dW_reg, expected)
    print("  ✓ L2 gradient correct")
    
    # Test gradient addition
    dW = np.array([[0.1, 0.2], [0.3, 0.4]])
    dW_total = apply_weight_decay_to_gradient(dW, W, weight_decay)
    expected = dW + weight_decay * W
    assert np.allclose(dW_total, expected)
    print("  ✓ Gradient augmentation correct")
    
    print("✓ All gradient tests passed!")


if __name__ == "__main__":
    print("="*60)
    print("Solution 3: Weight Decay (L2 Regularization)")
    print("="*60)
    
    demonstrate_weight_decay_effect()
    test_compute_l2_penalty()
    test_gradients()
    
    print("\n🎉 All tests passed!")
