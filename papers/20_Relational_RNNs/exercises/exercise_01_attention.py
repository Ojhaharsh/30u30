import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute 'Scaled Dot Product Attention'.
    
    Args:
        q (torch.Tensor): Queries (batch, heads, slots, head_dim)
        k (torch.Tensor): Keys (batch, heads, slots, head_dim)
        v (torch.Tensor): Values (batch, heads, slots, head_dim)
        mask (torch.Tensor): Optional mask (batch, 1, 1, slots)
        
    Returns:
        output (torch.Tensor): (batch, heads, slots, head_dim)
        attention_weights (torch.Tensor): (batch, heads, slots, slots)
    """
    d_k = q.size(-1)
    
    # TODO: Calculate attention scores
    # Hint: matmul(q, k.transpose(-2, -1)) / sqrt(d_k)
    scores = # YOUR CODE HERE
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # TODO: Apply softmax
    attn_weights = # YOUR CODE HERE
    
    # TODO: Weighted sum of values
    output = # YOUR CODE HERE
    
    return output, attn_weights

def test_attention():
    print("Testing Scaled Dot Product Attention...")
    batch, heads, slots, dim = 2, 4, 8, 16
    q = torch.randn(batch, heads, slots, dim)
    k = torch.randn(batch, heads, slots, dim)
    v = torch.randn(batch, heads, slots, dim)
    
    out, weights = scaled_dot_product_attention(q, k, v)
    
    assert out.shape == (batch, heads, slots, dim), f"Expected {(batch, heads, slots, dim)}, got {out.shape}"
    assert weights.shape == (batch, heads, slots, slots), f"Expected {(batch, heads, slots, slots)}, got {weights.shape}"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_attention()
