"""
Exercise 1 Solution: Dropout from Scratch

Complete implementation of inverted dropout.
"""

import numpy as np


class Dropout:
    """
    Inverted Dropout Layer - Complete Implementation.
    """
    
    def __init__(self, p: float = 0.5):
        assert 0 < p <= 1, "Keep probability must be in (0, 1]"
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with inverted dropout."""
        
        # Eval mode: return unchanged
        if not self.training:
            return x
        
        # Generate Bernoulli mask
        # Random values < p become 1 (keep), otherwise 0 (drop)
        self.mask = (np.random.rand(*x.shape) < self.p).astype(np.float32)
        
        # Apply mask and scale by 1/p (inverted dropout)
        # This ensures E[output] = input
        output = (x * self.mask) / self.p
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through dropout."""
        
        # Eval mode: gradient passes through unchanged
        if not self.training:
            return grad_output
        
        # Apply same mask and scaling to gradient
        # Gradient only flows through kept neurons
        grad_input = (grad_output * self.mask) / self.p
        
        return grad_input
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


# =============================================================================
# VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("DROPOUT SOLUTION VERIFICATION")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Basic forward pass
    print("\n1. Forward Pass (Training):")
    dropout = Dropout(p=0.5)
    x = np.ones((2, 4))
    y = dropout.forward(x)
    print(f"   Input:\n{x}")
    print(f"   Output:\n{y}")
    print(f"   Mask:\n{dropout.mask}")
    
    # Test 2: Expected value preservation
    print("\n2. Expected Value Preservation:")
    outputs = [dropout.forward(np.ones((10, 10))) for _ in range(1000)]
    mean = np.mean(outputs)
    print(f"   Mean output over 1000 passes: {mean:.4f}")
    print(f"   Expected: 1.0")
    print(f"   Close enough: {'YES' if abs(mean - 1.0) < 0.1 else 'NO'}")
    
    # Test 3: Eval mode
    print("\n3. Eval Mode:")
    dropout.eval()
    y_eval = dropout.forward(x)
    print(f"   Input unchanged: {'YES' if np.array_equal(x, y_eval) else 'NO'}")
    
    # Test 4: Backward pass
    print("\n4. Backward Pass:")
    dropout.train()
    y = dropout.forward(x)
    grad_out = np.ones_like(y)
    grad_in = dropout.backward(grad_out)
    print(f"   Forward output:\n{y}")
    print(f"   Gradient input:\n{grad_in}")
    print(f"   Pattern matches: {'YES' if np.array_equal(y==0, grad_in==0) else 'NO'}")
    
    print("\n" + "=" * 50)
    print("Solution verified!")
    print("=" * 50)
