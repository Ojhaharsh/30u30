import torch

def masked_aggregation(g_out: torch.Tensor) -> torch.Tensor:
    """
    TODO: Implement masked aggregation to ignore self-relations.
    
    In some relational tasks, we only care about relations between *different* objects
    (i != j). This function should mask out the diagonal elements of the relation matrix
    before summing.
    
    Input:
        g_out: Tensor of shape (batch, N, N, hidden_dim) representing g(o_i, o_j)
        
    Output:
        aggregated: Tensor of shape (batch, hidden_dim) = sum_{i != j} g(o_i, o_j)
        
    Constraints:
        - Use torch.eye to create a diagonal mask.
        - No for-loops.
    """
    # YOUR CODE HERE
    pass

def test_masking():
    batch, N, dim = 2, 3, 4
    # Create a tensor where diagonal elements are distinct
    # g_out[b, i, j]
    g_out = torch.ones(batch, N, N, dim)
    
    # Set diagonal to 100 to make it obvious if included
    for i in range(N):
        g_out[:, i, i, :] = 100.0
        
    # Expected sum per object pair (excluding diagonal)
    # Total pairs = N*N = 9. Diagonal = 3. Off-diagonal = 6.
    # Sum over N,N should be 6 * 1.0 = 6.0 per feature
    
    res = masked_aggregation(g_out)
    
    expected_value = (N * N - N) * 1.0
    
    if torch.allclose(res, torch.tensor(float(expected_value))):
        print("Exercise 6 (Part B): Masking [PASS]")
    else:
        print(f"Exercise 6 (Part B): [FAIL] Expected {expected_value}, got {res[0,0].item()}")
        print("Did you subtract/mask the diagonal (values=100)?")

if __name__ == "__main__":
    test_masking()
