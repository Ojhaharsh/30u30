import torch
import torch.nn as nn

def generate_pairs_via_broadcasting(objects: torch.Tensor) -> torch.Tensor:
    """
    TODO: Implement pairwise object concatenation using broadcasting.
    As described in Section 2.1, the RN computes relations over all pairs.
    
    Input: (batch, N, D)
    Output: (batch, N^2, 2*D)
    
    Constraint: No Python for-loops. Use unsqueeze, expand/repeat, and cat.
    """
    # YOUR CODE HERE
    pass

def test_pair_generator():
    batch, N, D = 2, 5, 8
    x = torch.randn(batch, N, D)
    
    pairs = generate_pairs_via_broadcasting(x)
    
    # Validation
    assert pairs.shape == (batch, N*N, 2*D), f"Expected {(batch, N*N, 2*D)}, got {pairs.shape}"
    
    # Check if first pair is actually (o0, o0)
    first_pair = pairs[0, 0]
    expected_first = torch.cat([x[0, 0], x[0, 0]], dim=0)
    assert torch.allclose(first_pair, expected_first), "First pair is incorrect."
    
    # Check if second pair is (o0, o1)
    second_pair = pairs[0, 1]
    expected_second = torch.cat([x[0, 0], x[0, 1]], dim=0)
    assert torch.allclose(second_pair, expected_second), "Second pair is incorrect."
    
    print("Exercise 1: [PASS]")

if __name__ == "__main__":
    test_pair_generator()
