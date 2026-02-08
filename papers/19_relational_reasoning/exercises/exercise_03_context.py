import torch
from implementation import RelationNetwork

def forward_with_context(model: RelationNetwork, objects: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """
    TODO: Implement the forward pass with question conditioning.
    As described in Section 2.1, the relation function g_theta can take 
    an optional context vector 'q'.
    
    1. Use model.generate_pairs(objects, question) to get (B, N^2, 2*D + Q).
    2. Apply model.g_theta.
    3. Aggregate and reasoning via f_phi.
    """
    # YOUR CODE HERE
    pass

def test_context_injection():
    obj_dim, q_dim = 4, 8
    model = RelationNetwork(object_dim=obj_dim, question_dim=q_dim, output_dim=2)
    
    objects = torch.randn(1, 5, obj_dim)
    question = torch.randn(1, q_dim)
    
    out = forward_with_context(model, objects, question)
    assert out.shape == (1, 2)
    print("Exercise 3: [PASS]")

if __name__ == "__main__":
    print("-" * 50)
    print("Exercise 3: Question Context Injection")
    print("-" * 50)
    test_context_injection()
