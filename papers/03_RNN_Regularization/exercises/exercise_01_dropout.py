"""
Exercise 1: Dropout Layer
=========================

Goal: Implement dropout forward and backward passes from scratch.

Your Task:
- Fill in the TODOs below to complete the dropout implementation
- Test your implementation with the provided tests
- Compare with the reference solution

Learning Objectives:
1. Understand how dropout randomly disables neurons
2. Learn why we scale by 1/keep_prob (inverted dropout)
3. See the difference between training and inference modes
4. Understand how gradients flow through dropout

Time: 30-45 minutes
Difficulty: Medium ⏱️⏱️
"""

import numpy as np


def dropout_forward(x: np.ndarray, keep_prob: float, training: bool = True):
    """
    Apply dropout to input activations.
    
    Args:
        x: Input activations, shape (any)
        keep_prob: Probability of KEEPING each neuron (0.8 = 20% dropout)
        training: Whether we're in training mode
        
    Returns:
        out: Output after dropout
        mask: Dropout mask (needed for backward pass)
        
    Example:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        keep_prob = 0.8
        
        Training output might be: [1.25, 2.5, 0.0, 5.0, 6.25]
                                   ↑     ↑    ↑    ↑    ↑
                                   kept  kept drop kept kept
                                   (all scaled by 1/0.8 = 1.25)
        
        Test output: [1.0, 2.0, 3.0, 4.0, 5.0] (unchanged)
    """
    # TODO 1: Handle the case when we're NOT training
    # If training=False or keep_prob=1.0, return x unchanged and mask=None
    # Hint: if not training or keep_prob == 1.0:
    #           return ???, ???
    
    # TODO 2: Create a binary mask
    # Use np.random.binomial(1, keep_prob, size=x.shape)
    # This creates 1s with probability keep_prob, 0s otherwise
    # mask = ???
    
    # TODO 3: Scale the mask by 1/keep_prob
    # This is "inverted dropout" - scale during training so test time is unchanged
    # Why? E[mask * x] = keep_prob * x, so we divide by keep_prob to get E[out] = x
    # mask = mask / ???
    
    # TODO 4: Apply mask to input
    # out = ???
    
    # TODO 5: Return output and mask
    # return ???, ???
    
    pass  # Remove this when you implement


def dropout_backward(dout: np.ndarray, mask: np.ndarray):
    """
    Backward pass for dropout.
    
    Args:
        dout: Gradient of loss with respect to dropout output
        mask: Mask from forward pass (already scaled by 1/keep_prob)
        
    Returns:
        dx: Gradient of loss with respect to dropout input
        
    Key insight:
        Gradients only flow through neurons that were KEPT.
        Since mask is already scaled, we just multiply: dx = dout * mask
    """
    # TODO 6: Handle the case when mask is None
    # If mask is None, no dropout was applied, so gradient passes through unchanged
    # if mask is None:
    #     return ???
    
    # TODO 7: Apply mask to gradient
    # Gradients flow through the same neurons that activations flowed through
    # dx = ???
    
    # TODO 8: Return gradient
    # return ???
    
    pass  # Remove this when you implement


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_dropout_forward():
    """Test the dropout forward pass."""
    print("Testing dropout_forward...")
    
    # Test 1: No dropout mode (training=False)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out, mask = dropout_forward(x, keep_prob=1.0, training=False)
    assert np.allclose(out, x), "Should return input unchanged when training=False"
    assert mask is None, "Should return None mask when training=False"
    print("  ✓ Test 1: No dropout mode works")
    
    # Test 2: Shape preservation
    np.random.seed(42)
    x = np.random.randn(100)
    out, mask = dropout_forward(x, keep_prob=0.8, training=True)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert mask.shape == x.shape, f"Mask shape mismatch: {mask.shape} vs {x.shape}"
    print("  ✓ Test 2: Shape preservation works")
    
    # Test 3: Correct dropout rate
    np.random.seed(42)
    x = np.ones(10000)
    out, mask = dropout_forward(x, keep_prob=0.8, training=True)
    dropout_rate = np.mean(mask == 0)
    assert 0.15 < dropout_rate < 0.25, f"Dropout rate should be ~20%, got {dropout_rate*100:.1f}%"
    print(f"  ✓ Test 3: Dropout rate correct (~20%, got {dropout_rate*100:.1f}%)")
    
    # Test 4: Correct scaling
    np.random.seed(42)
    x = np.ones(10000)
    out, mask = dropout_forward(x, keep_prob=0.5, training=True)
    active_values = out[out > 0]
    assert 1.9 < np.mean(active_values) < 2.1, f"Active values should be ~2.0, got {np.mean(active_values):.2f}"
    print(f"  ✓ Test 4: Output scaling correct (active mean={np.mean(active_values):.2f})")
    
    # Test 5: Expected value preservation
    np.random.seed(42)
    x = np.ones(100000) * 5.0
    out, mask = dropout_forward(x, keep_prob=0.8, training=True)
    assert 4.9 < np.mean(out) < 5.1, f"Expected value should be ~5.0, got {np.mean(out):.2f}"
    print(f"  ✓ Test 5: Expected value preserved (mean={np.mean(out):.2f})")
    
    print("✓ All dropout_forward tests passed!\n")


def test_dropout_backward():
    """Test the dropout backward pass."""
    print("Testing dropout_backward...")
    
    # Test 1: No mask case
    dout = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dx = dropout_backward(dout, mask=None)
    assert np.allclose(dx, dout), "Should return dout unchanged when mask is None"
    print("  ✓ Test 1: No mask case works")
    
    # Test 2: Gradient masking
    dout = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([1.25, 0.0, 1.25, 0.0, 1.25])  # Scaled mask
    dx = dropout_backward(dout, mask)
    expected = dout * mask
    assert np.allclose(dx, expected), f"Gradient should be dout * mask"
    print("  ✓ Test 2: Gradient masking works")
    
    # Test 3: Zeros propagate correctly
    dout = np.ones(100)
    mask = np.zeros(100)
    mask[::2] = 2.0  # Every other element is active
    dx = dropout_backward(dout, mask)
    assert np.sum(dx[1::2]) == 0, "Gradient should be zero where mask is zero"
    print("  ✓ Test 3: Zero gradients propagate correctly")
    
    print("✓ All dropout_backward tests passed!\n")


def test_dropout_integration():
    """Test forward and backward together."""
    print("Testing dropout integration...")
    
    # Do a forward pass, then backward pass
    np.random.seed(42)
    x = np.random.randn(100)
    
    # Forward
    out, mask = dropout_forward(x, keep_prob=0.7, training=True)
    
    # Backward (pretend gradient from next layer is all ones)
    dout = np.ones_like(out)
    dx = dropout_backward(dout, mask)
    
    # Check that gradient is zero where output was zero
    zeros_out = out == 0
    zeros_dx = dx == 0
    assert np.allclose(zeros_out, zeros_dx), "Gradient should be zero where output was zero"
    print("  ✓ Forward and backward are consistent")
    
    print("✓ Integration test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 1: Dropout Layer")
    print("="*60 + "\n")
    
    try:
        test_dropout_forward()
        test_dropout_backward()
        test_dropout_integration()
        print("🎉 All tests passed! You've mastered dropout!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
