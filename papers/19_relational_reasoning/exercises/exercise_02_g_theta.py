import torch
import torch.nn as nn

class RelationFunction(nn.Module):
    """
    TODO: Implement the 'g_theta' MLP.
    
    As described in Section 2, g_theta is a multi-layer perceptron that processes
    each pair of objects (and optionally a question embedding).
    
    Architecture:
    - Input: (2 * object_dim) + question_dim
    - Hidden: 4 layers of 256 units with ReLU
    - Output: 256 units
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

def test_relation_function():
    input_dim = 16
    hidden_dim = 256
    model = RelationFunction(input_dim, hidden_dim)
    
    # Test random input
    x = torch.randn(10, input_dim)
    try:
        out = model(x)
        assert out.shape == (10, hidden_dim), f"Expected {(10, hidden_dim)}, got {out.shape}"
        print("Exercise 2 (Part A): Relation Function [PASS]")
    except Exception as e:
        print(f"Exercise 2 (Part A): [FAIL] {e}")

if __name__ == "__main__":
    test_relation_function()
