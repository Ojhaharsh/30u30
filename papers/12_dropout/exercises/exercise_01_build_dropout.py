"""
Exercise 1: Build Dropout from Scratch

Implement a complete Dropout layer with forward and backward passes.
Uses inverted dropout (scale during training, not testing).
"""

import numpy as np


class Dropout:
    """
    Inverted Dropout Layer.
    
    TODO: Implement the forward and backward methods.
    
    Args:
        p: Probability of KEEPING a neuron (not dropping!)
           p=0.5 means 50% of neurons are kept.
    """
    
    def __init__(self, p: float = 0.5):
        assert 0 < p <= 1, "Keep probability must be in (0, 1]"
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with inverted dropout.
        
        During training:
        1. Generate random mask: 1 with probability p, 0 with probability (1-p)
        2. Apply mask to input: x * mask
        3. Scale by 1/p to maintain expected value
        
        During inference:
        - Return input unchanged (scaling was done during training)
        
        Args:
            x: Input array of any shape
            
        Returns:
            Dropped and scaled output (same shape as input)
            
        Hint:
        - Use np.random.rand(*x.shape) to get uniform random values
        - Compare with self.p to get binary mask
        - Store mask for backward pass
        """
        # ==========================================
        # TODO: Implement forward pass
        # ==========================================
        
        # Step 1: Check if in eval mode
        # if not self.training:
        #     return ???
        
        # Step 2: Generate Bernoulli mask
        # self.mask = ???
        
        # Step 3: Apply mask and scale
        # output = ???
        
        # return output
        
        raise NotImplementedError("Implement the forward pass!")
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through dropout.
        
        The gradient only flows through neurons that were kept.
        Same mask and scaling factor are applied.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
            
        Hint:
        - Use the cached self.mask from forward pass
        - Apply same scaling factor as forward
        """
        # ==========================================
        # TODO: Implement backward pass
        # ==========================================
        
        # Step 1: Check if in eval mode
        # if not self.training:
        #     return ???
        
        # Step 2: Apply mask and scaling to gradient
        # grad_input = ???
        
        # return grad_input
        
        raise NotImplementedError("Implement the backward pass!")
    
    def train(self):
        """Set to training mode (dropout enabled)."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode (dropout disabled)."""
        self.training = False


# =============================================================================
# TESTS
# =============================================================================

def test_forward_training():
    """Test forward pass in training mode."""
    np.random.seed(42)
    
    dropout = Dropout(p=0.5)
    dropout.training = True
    
    x = np.ones((4, 8))
    y = dropout.forward(x)
    
    # Some values should be 0 (dropped)
    assert np.any(y == 0), "Some values should be dropped (equal to 0)"
    
    # Non-zero values should be scaled by 1/p = 2
    non_zero = y[y != 0]
    assert np.allclose(non_zero, 2.0), f"Kept values should be 2.0, got {np.unique(non_zero)}"
    
    # Approximately half should be kept
    keep_ratio = np.mean(y != 0)
    assert 0.3 < keep_ratio < 0.7, f"Keep ratio should be ~0.5, got {keep_ratio}"
    
    print("[PASS] Forward training mode test passed!")


def test_forward_eval():
    """Test forward pass in evaluation mode."""
    dropout = Dropout(p=0.5)
    dropout.training = False
    
    x = np.ones((4, 8)) * 3.14
    y = dropout.forward(x)
    
    # In eval mode, output should equal input exactly
    assert np.array_equal(y, x), "Eval mode should return unchanged input"
    
    print("[PASS] Forward eval mode test passed!")


def test_expected_value():
    """Test that expected value is preserved."""
    np.random.seed(42)
    
    dropout = Dropout(p=0.5)
    dropout.training = True
    
    x = np.ones((10, 10)) * 5.0
    
    # Run many forward passes
    outputs = [dropout.forward(x.copy()) for _ in range(1000)]
    mean_output = np.mean(outputs, axis=0)
    
    # Expected value should be close to original
    assert np.allclose(mean_output, x, atol=0.2), \
        f"Expected output mean ~{x[0,0]}, got {mean_output.mean():.2f}"
    
    print("[PASS] Expected value test passed!")


def test_backward():
    """Test backward pass."""
    np.random.seed(42)
    
    dropout = Dropout(p=0.5)
    dropout.training = True
    
    x = np.ones((4, 8))
    y = dropout.forward(x)
    
    # Backward pass
    grad_output = np.ones_like(y)
    grad_input = dropout.backward(grad_output)
    
    # Gradient should have same shape
    assert grad_input.shape == x.shape, "Gradient shape mismatch"
    
    # Gradient should be 0 where forward was 0, and 2 where forward was 2
    assert np.all((grad_input == 0) | (grad_input == 2)), \
        "Gradient values should be 0 or 2 (scaled)"
    
    # Gradient pattern should match forward pattern
    assert np.array_equal(grad_input == 0, y == 0), \
        "Gradient zeros should match forward zeros"
    
    print("[PASS] Backward pass test passed!")


def test_backward_eval():
    """Test backward pass in eval mode."""
    dropout = Dropout(p=0.5)
    dropout.training = False
    
    x = np.ones((4, 8))
    y = dropout.forward(x)
    
    grad_output = np.ones_like(y) * 0.5
    grad_input = dropout.backward(grad_output)
    
    # In eval mode, gradient should pass through unchanged
    assert np.array_equal(grad_input, grad_output), \
        "Eval mode gradient should pass through unchanged"
    
    print("[PASS] Backward eval mode test passed!")


def test_numerical_gradient():
    """Numerical gradient check."""
    np.random.seed(42)
    
    dropout = Dropout(p=0.7)
    dropout.training = True
    
    x = np.random.randn(2, 4)
    
    # Forward pass
    y = dropout.forward(x)
    
    # Analytical gradient
    grad_output = np.ones_like(y)
    analytical_grad = dropout.backward(grad_output)
    
    # Numerical gradient
    eps = 1e-5
    numerical_grad = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Use same mask for both perturbations
            mask = dropout.mask
            
            x_plus = x.copy()
            x_plus[i, j] += eps
            y_plus = (x_plus * mask) / dropout.p
            
            x_minus = x.copy()  
            x_minus[i, j] -= eps
            y_minus = (x_minus * mask) / dropout.p
            
            numerical_grad[i, j] = (y_plus.sum() - y_minus.sum()) / (2 * eps)
    
    # Compare
    diff = np.abs(analytical_grad - numerical_grad).max()
    assert diff < 1e-5, f"Numerical gradient check failed, max diff: {diff}"
    
    print("[PASS] Numerical gradient test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("TESTING DROPOUT IMPLEMENTATION")
    print("=" * 50)
    
    try:
        test_forward_training()
        test_forward_eval()
        test_expected_value()
        test_backward()
        test_backward_eval()
        test_numerical_gradient()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        
    except NotImplementedError as e:
        print(f"\n[ERROR] {e}")
        print("Implement the TODO sections and run again!")
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        print("Check your implementation and try again!")


if __name__ == "__main__":
    run_all_tests()
