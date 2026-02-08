import torch

def create_contextual_pairs(objects: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """
    TODO: Implement the pairing logic with Question Injection.
    
    In Sort-of-CLEVR (Section 4.1), the relation g_theta(o_i, o_j, q) depends 
    on the question 'q'. We need to broadcast 'q' to every pair.
    
    Inputs:
        objects: (batch_size, N, object_dim)
        question: (batch_size, question_dim)
        
    Output:
        pairs: (batch_size, N, N, 2*object_dim + question_dim)
        
    Hint:
        1. Create pairs (o_i, o_j) of shape (B, N, N, 2*O)
        2. Expand question 'q' to (B, N, N, Q)
        3. Concatenate along the last dimension.
    """
    # YOUR CODE HERE
    pass

def test_contextual_pairs():
    # Setup
    batch, N, obj_dim, q_dim = 2, 3, 4, 5
    objects = torch.randn(batch, N, obj_dim)
    question = torch.randn(batch, q_dim)
    
    try:
        pairs = create_contextual_pairs(objects, question)
        
        # Check shape
        expected_shape = (batch, N, N, 2 * obj_dim + q_dim)
        assert pairs.shape == expected_shape, f"Expected {expected_shape}, got {pairs.shape}"
        
        # Check content of first pair (0,0) with question
        # pair[0,0,0] should be cat(avg_obj[0,0], avg_obj[0,0], q[0])
        # But let's just check the question part is correct
        q_part = pairs[:, :, :, -q_dim:]
        
        # Check if question vector is correctly broadcast
        # q_part[b, i, j] should equal question[b]
        diff = (q_part - question.view(batch, 1, 1, q_dim)).abs().max()
        assert diff < 1e-5, "Question vector was not broadcast correctly."
        
        print("Exercise 4: Contextual Pairs [PASS]")
        
    except Exception as e:
        print(f"Exercise 4: [FAIL] {e}")

if __name__ == "__main__":
    test_contextual_pairs()
