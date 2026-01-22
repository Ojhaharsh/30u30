"""
Exercise 2: Layer Normalization
===============================

Goal: Implement layer normalization forward and backward passes from scratch.

Your Task:
- Fill in the TODOs below to complete the layer norm implementation
- Test your implementation with the provided tests
- Compare with the reference solution

Learning Objectives:
1. Understand how normalization stabilizes training
2. Learn to compute per-sample statistics (not batch statistics)
3. Implement learnable scale (gamma) and shift (beta) parameters
4. Derive and implement the backward pass

Time: 45-60 minutes
Difficulty: Hard ⏱️⏱️⏱️

Key Equation:
    y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
"""

import numpy as np


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """
    Layer normalization forward pass.
    
    Args:
        x: Input, shape (batch_size, features) or (features,)
        gamma: Scale parameter, shape (features,)
        beta: Shift parameter, shape (features,)
        eps: Small constant for numerical stability
        
    Returns:
        out: Normalized output, same shape as x
        cache: Values needed for backward pass
        
    Layer norm normalizes across the LAST dimension (features).
    This is different from batch norm which normalizes across the batch.
    
    Example:
        x = [[1, 2, 3],      # Sample 1
             [4, 5, 6]]      # Sample 2
        
        After layer norm, EACH ROW has mean=0 and variance=1:
        out = [[-1.22, 0, 1.22],    # Normalized sample 1
               [-1.22, 0, 1.22]]    # Normalized sample 2
    """
    # TODO 1: Compute mean across the last dimension
    # Use keepdims=True to preserve shape for broadcasting
    # mean = np.mean(x, axis=???, keepdims=???)
    mean = None  # TODO: Implement
    
    # TODO 2: Compute variance across the last dimension
    # var = np.var(x, axis=???, keepdims=???)
    var = None  # TODO: Implement
    
    # TODO 3: Normalize: x_hat = (x - mean) / sqrt(var + eps)
    # The eps prevents division by zero when variance is very small
    std = None  # TODO: sqrt(var + eps)
    x_hat = None  # TODO: (x - mean) / std
    
    # TODO 4: Scale and shift: out = gamma * x_hat + beta
    # gamma and beta are learnable parameters
    out = None  # TODO: Implement
    
    # TODO 5: Store values needed for backward pass
    cache = (x, x_hat, mean, var, std, gamma, beta, eps)
    
    # TODO 6: Return output and cache
    # return ???, cache
    
    pass  # Remove this when you implement


def layer_norm_backward(dout: np.ndarray, cache: tuple):
    """
    Layer normalization backward pass.
    
    Args:
        dout: Gradient of loss with respect to output, same shape as x
        cache: Values from forward pass
        
    Returns:
        dx: Gradient with respect to input x
        dgamma: Gradient with respect to gamma
        dbeta: Gradient with respect to beta
        
    The backward pass is complex because x_hat depends on the mean and variance,
    which in turn depend on all elements of x.
    """
    x, x_hat, mean, var, std, gamma, beta, eps = cache
    
    # Get the number of features (last dimension)
    N = x.shape[-1] if x.ndim > 1 else x.shape[0]
    
    # TODO 7: Gradient with respect to beta
    # dbeta = sum of dout (across batch dimension if present)
    # Hint: np.sum(dout, axis=0) if dout is 2D, else just dout
    dbeta = None  # TODO: Implement
    
    # TODO 8: Gradient with respect to gamma
    # dgamma = sum of (dout * x_hat)
    dgamma = None  # TODO: Implement
    
    # TODO 9: Gradient with respect to x_hat
    # dx_hat = dout * gamma
    dx_hat = None  # TODO: Implement
    
    # TODO 10: Gradient with respect to x (the tricky part!)
    # This requires the chain rule through mean and variance
    # 
    # dx = (1/N) * (1/std) * (N * dx_hat 
    #                        - sum(dx_hat) 
    #                        - x_hat * sum(dx_hat * x_hat))
    #
    # The sums are across the last dimension (features)
    dx = None  # TODO: Implement this formula
    
    # Hint for TODO 10:
    # Step 1: Compute sum(dx_hat, axis=-1, keepdims=True)
    # Step 2: Compute sum(dx_hat * x_hat, axis=-1, keepdims=True)
    # Step 3: Combine using the formula above
    
    # TODO 11: Return gradients
    # return dx, dgamma, dbeta
    
    pass  # Remove this when you implement


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_layer_norm_forward():
    """Test the layer normalization forward pass."""
    print("Testing layer_norm_forward...")
    
    # Test 1: Basic normalization (1D)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gamma = np.ones(5)
    beta = np.zeros(5)
    out, cache = layer_norm_forward(x, gamma, beta)
    
    # Check mean is ~0
    assert np.abs(np.mean(out)) < 1e-6, f"Mean should be ~0, got {np.mean(out)}"
    # Check std is ~1
    assert np.abs(np.std(out) - 1) < 1e-6, f"Std should be ~1, got {np.std(out)}"
    print("  ✓ Test 1: Basic normalization works (mean=0, std=1)")
    
    # Test 2: Batch normalization (2D)
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    gamma = np.ones(3)
    beta = np.zeros(3)
    out, cache = layer_norm_forward(x, gamma, beta)
    
    # Check each row is normalized
    for i in range(2):
        row_mean = np.mean(out[i])
        row_std = np.std(out[i])
        assert np.abs(row_mean) < 1e-6, f"Row {i} mean should be ~0, got {row_mean}"
        assert np.abs(row_std - 1) < 0.1, f"Row {i} std should be ~1, got {row_std}"
    print("  ✓ Test 2: Batch normalization works (each row normalized)")
    
    # Test 3: Scale and shift
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gamma = np.ones(5) * 2.0  # Scale by 2
    beta = np.ones(5) * 3.0   # Shift by 3
    out, cache = layer_norm_forward(x, gamma, beta)
    
    # After normalization: mean=0, std=1
    # After scale by 2: mean=0, std=2
    # After shift by 3: mean=3, std=2
    assert np.abs(np.mean(out) - 3.0) < 1e-6, f"Mean should be ~3, got {np.mean(out)}"
    assert np.abs(np.std(out) - 2.0) < 0.1, f"Std should be ~2, got {np.std(out)}"
    print("  ✓ Test 3: Scale and shift work correctly")
    
    # Test 4: Constant input
    x = np.ones(5) * 5.0
    gamma = np.ones(5)
    beta = np.zeros(5)
    out, cache = layer_norm_forward(x, gamma, beta, eps=1e-5)
    
    # Constant input has variance 0, so output should be all zeros (before scale/shift)
    # With eps, this should not produce NaN
    assert not np.any(np.isnan(out)), "Should not produce NaN for constant input"
    print("  ✓ Test 4: Constant input handled (no NaN)")
    
    print("✓ All layer_norm_forward tests passed!\n")


def test_layer_norm_backward():
    """Test the layer normalization backward pass with numerical gradient checking."""
    print("Testing layer_norm_backward...")
    
    # Set up test case
    np.random.seed(42)
    x = np.random.randn(3, 4)  # Batch of 3, 4 features each
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    
    # Forward pass
    out, cache = layer_norm_forward(x, gamma, beta)
    
    # Backward pass
    dout = np.random.randn(*out.shape)
    dx, dgamma, dbeta = layer_norm_backward(dout, cache)
    
    # Test 1: Shapes
    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape} vs {x.shape}"
    assert dgamma.shape == gamma.shape, f"dgamma shape mismatch: {dgamma.shape} vs {gamma.shape}"
    assert dbeta.shape == beta.shape, f"dbeta shape mismatch: {dbeta.shape} vs {beta.shape}"
    print("  ✓ Test 1: Gradient shapes are correct")
    
    # Test 2: Numerical gradient check for dbeta
    eps_num = 1e-5
    dbeta_num = np.zeros_like(beta)
    for i in range(len(beta)):
        beta_plus = beta.copy()
        beta_plus[i] += eps_num
        out_plus, _ = layer_norm_forward(x, gamma, beta_plus)
        
        beta_minus = beta.copy()
        beta_minus[i] -= eps_num
        out_minus, _ = layer_norm_forward(x, gamma, beta_minus)
        
        dbeta_num[i] = np.sum(dout * (out_plus - out_minus) / (2 * eps_num))
    
    dbeta_error = np.max(np.abs(dbeta - dbeta_num))
    assert dbeta_error < 1e-4, f"dbeta error too large: {dbeta_error}"
    print(f"  ✓ Test 2: dbeta numerical gradient check passed (error={dbeta_error:.2e})")
    
    # Test 3: Numerical gradient check for dgamma
    dgamma_num = np.zeros_like(gamma)
    for i in range(len(gamma)):
        gamma_plus = gamma.copy()
        gamma_plus[i] += eps_num
        out_plus, _ = layer_norm_forward(x, gamma_plus, beta)
        
        gamma_minus = gamma.copy()
        gamma_minus[i] -= eps_num
        out_minus, _ = layer_norm_forward(x, gamma_minus, beta)
        
        dgamma_num[i] = np.sum(dout * (out_plus - out_minus) / (2 * eps_num))
    
    dgamma_error = np.max(np.abs(dgamma - dgamma_num))
    assert dgamma_error < 1e-4, f"dgamma error too large: {dgamma_error}"
    print(f"  ✓ Test 3: dgamma numerical gradient check passed (error={dgamma_error:.2e})")
    
    print("✓ All layer_norm_backward tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 2: Layer Normalization")
    print("="*60 + "\n")
    
    try:
        test_layer_norm_forward()
        test_layer_norm_backward()
        print("🎉 All tests passed! You've mastered layer normalization!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
