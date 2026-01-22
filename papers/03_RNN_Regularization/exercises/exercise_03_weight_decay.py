"""
Exercise 3: Weight Decay (L2 Regularization)
=============================================

Goal: Implement L2 regularization that penalizes large weights.

Your Task:
- Fill in the TODOs below to complete the weight decay implementation
- Test your implementation with the provided tests
- Compare with the reference solution

Learning Objectives:
1. Understand why simpler models (smaller weights) generalize better
2. Implement the L2 penalty term
3. See how weight decay affects the loss function
4. Understand the gradient contribution from weight decay

Time: 20-30 minutes
Difficulty: Easy ⏱️

Key Equation:
    L_total = L_model + (λ/2) * Σ(w²)
    
    Where:
    - L_model is your normal loss (cross-entropy, MSE, etc.)
    - λ is the weight decay coefficient
    - Σ(w²) is the sum of squared weights
    
    The gradient contribution is: dW += λ * W
"""

import numpy as np


def compute_l2_penalty(weights_list: list, weight_decay: float) -> float:
    """
    Compute L2 regularization penalty.
    
    Args:
        weights_list: List of weight matrices [W1, W2, W3, ...]
        weight_decay: Regularization coefficient (lambda, λ)
        
    Returns:
        penalty: Scalar penalty value
        
    Formula:
        penalty = (λ/2) * Σ(w²) for all weights w in all matrices
        
    Why the 1/2?
        Makes the gradient cleaner: d/dw[(λ/2)*w²] = λ*w
        Without 1/2: d/dw[λ*w²] = 2*λ*w (extra factor of 2)
        
    Example:
        W1 = [[1, 2], [3, 4]]  # Sum of squares = 1+4+9+16 = 30
        W2 = [[1, 1]]          # Sum of squares = 1+1 = 2
        weight_decay = 0.01
        
        penalty = 0.01/2 * (30 + 2) = 0.16
    """
    # TODO 1: Initialize penalty to 0
    penalty = 0.0
    
    # TODO 2: Loop through each weight matrix
    # For each matrix, add the sum of squared elements
    # for W in weights_list:
    #     penalty += ???
    
    # TODO 3: Multiply by weight_decay/2
    # penalty = (weight_decay / 2) * penalty
    
    # TODO 4: Return penalty
    # return ???
    
    pass  # Remove this when you implement


def compute_l2_gradient(W: np.ndarray, weight_decay: float) -> np.ndarray:
    """
    Compute the gradient contribution from L2 regularization.
    
    Args:
        W: Weight matrix
        weight_decay: Regularization coefficient (lambda, λ)
        
    Returns:
        dW_reg: Gradient contribution from regularization
        
    Formula:
        The penalty is (λ/2) * Σ(w²)
        The gradient with respect to w is: d/dw[(λ/2)*w²] = λ*w
        
    So the gradient contribution is simply: λ * W
    """
    # TODO 5: Compute gradient contribution
    # dW_reg = weight_decay * W
    # return dW_reg
    
    pass  # Remove this when you implement


def regularized_loss(model_loss: float, weights_list: list, weight_decay: float) -> float:
    """
    Compute total loss with L2 regularization.
    
    Args:
        model_loss: Loss from the model (cross-entropy, MSE, etc.)
        weights_list: List of weight matrices
        weight_decay: Regularization coefficient
        
    Returns:
        total_loss: model_loss + L2 penalty
        
    Example:
        model_loss = 2.5
        weights = [array of weights]
        weight_decay = 0.0001
        
        penalty = 0.0001/2 * sum(w²) = 0.05
        total_loss = 2.5 + 0.05 = 2.55
    """
    # TODO 6: Compute L2 penalty using compute_l2_penalty
    # penalty = ???
    
    # TODO 7: Return total loss
    # return model_loss + penalty
    
    pass  # Remove this when you implement


def apply_weight_decay_to_gradient(dW: np.ndarray, W: np.ndarray, weight_decay: float) -> np.ndarray:
    """
    Add weight decay contribution to gradient.
    
    Args:
        dW: Gradient from backpropagation
        W: Weight matrix
        weight_decay: Regularization coefficient
        
    Returns:
        dW_total: dW + weight_decay * W
        
    This is what happens during optimization:
        1. Compute gradient from loss: dW = d(loss)/dW
        2. Add regularization gradient: dW += λ * W
        3. Update weights: W = W - lr * dW
        
    The effect: Weights are pulled toward zero by the amount λ * W
    Large weights are pulled more strongly than small weights.
    """
    # TODO 8: Add weight decay to gradient
    # dW_total = dW + weight_decay * W
    # return dW_total
    
    pass  # Remove this when you implement


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_compute_l2_penalty():
    """Test L2 penalty computation."""
    print("Testing compute_l2_penalty...")
    
    # Test 1: Simple case
    W1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # Sum of squares = 1+4+9+16 = 30
    penalty = compute_l2_penalty([W1], weight_decay=0.1)
    expected = 0.1 / 2 * 30  # = 1.5
    assert np.abs(penalty - expected) < 1e-6, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Test 1: Simple case works (penalty={penalty:.2f})")
    
    # Test 2: Multiple weight matrices
    W1 = np.ones((2, 2))  # Sum = 4
    W2 = np.ones((3, 3))  # Sum = 9
    penalty = compute_l2_penalty([W1, W2], weight_decay=0.01)
    expected = 0.01 / 2 * (4 + 9)  # = 0.065
    assert np.abs(penalty - expected) < 1e-6, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Test 2: Multiple matrices work (penalty={penalty:.4f})")
    
    # Test 3: Zero weight decay
    W1 = np.random.randn(10, 10)
    penalty = compute_l2_penalty([W1], weight_decay=0.0)
    assert penalty == 0.0, f"Zero weight decay should give zero penalty"
    print("  ✓ Test 3: Zero weight decay works")
    
    # Test 4: Empty weights list
    penalty = compute_l2_penalty([], weight_decay=0.1)
    assert penalty == 0.0, f"Empty list should give zero penalty"
    print("  ✓ Test 4: Empty list works")
    
    print("✓ All compute_l2_penalty tests passed!\n")


def test_compute_l2_gradient():
    """Test L2 gradient computation."""
    print("Testing compute_l2_gradient...")
    
    # Test 1: Gradient should be λ * W
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight_decay = 0.1
    dW_reg = compute_l2_gradient(W, weight_decay)
    expected = weight_decay * W
    assert np.allclose(dW_reg, expected), f"Gradient mismatch"
    print("  ✓ Test 1: Gradient = λ * W")
    
    # Test 2: Zero weight decay
    W = np.random.randn(5, 5)
    dW_reg = compute_l2_gradient(W, weight_decay=0.0)
    assert np.allclose(dW_reg, np.zeros_like(W)), "Zero decay should give zero gradient"
    print("  ✓ Test 2: Zero weight decay works")
    
    # Test 3: Shape preservation
    W = np.random.randn(10, 20)
    dW_reg = compute_l2_gradient(W, weight_decay=0.01)
    assert dW_reg.shape == W.shape, f"Shape mismatch: {dW_reg.shape} vs {W.shape}"
    print("  ✓ Test 3: Shape preserved")
    
    print("✓ All compute_l2_gradient tests passed!\n")


def test_regularized_loss():
    """Test regularized loss computation."""
    print("Testing regularized_loss...")
    
    # Test 1: Basic case
    model_loss = 2.5
    W = np.ones((4, 4))  # Sum of squares = 16
    total = regularized_loss(model_loss, [W], weight_decay=0.1)
    expected = 2.5 + 0.1 / 2 * 16  # = 2.5 + 0.8 = 3.3
    assert np.abs(total - expected) < 1e-6, f"Expected {expected}, got {total}"
    print(f"  ✓ Test 1: Basic case works (total={total:.2f})")
    
    # Test 2: Zero weight decay should return model loss
    model_loss = 5.0
    W = np.random.randn(10, 10)
    total = regularized_loss(model_loss, [W], weight_decay=0.0)
    assert total == model_loss, f"Zero decay should return model loss"
    print("  ✓ Test 2: Zero weight decay works")
    
    print("✓ All regularized_loss tests passed!\n")


def test_apply_weight_decay_to_gradient():
    """Test weight decay gradient application."""
    print("Testing apply_weight_decay_to_gradient...")
    
    # Test 1: Basic case
    dW = np.array([[0.1, 0.2], [0.3, 0.4]])
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight_decay = 0.1
    dW_total = apply_weight_decay_to_gradient(dW, W, weight_decay)
    expected = dW + weight_decay * W
    assert np.allclose(dW_total, expected), f"Gradient mismatch"
    print("  ✓ Test 1: Basic case works")
    
    # Test 2: Zero weight decay should return original gradient
    dW = np.random.randn(5, 5)
    W = np.random.randn(5, 5)
    dW_total = apply_weight_decay_to_gradient(dW, W, weight_decay=0.0)
    assert np.allclose(dW_total, dW), "Zero decay should return original gradient"
    print("  ✓ Test 2: Zero weight decay works")
    
    print("✓ All apply_weight_decay_to_gradient tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 3: Weight Decay (L2 Regularization)")
    print("="*60 + "\n")
    
    try:
        test_compute_l2_penalty()
        test_compute_l2_gradient()
        test_regularized_loss()
        test_apply_weight_decay_to_gradient()
        print("🎉 All tests passed! You've mastered weight decay!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
