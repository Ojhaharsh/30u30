"""
Solution 2: Order-Invariant Set Encoder

Complete implementation with permutation invariance tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """
    Order-invariant encoder using self-attention WITHOUT positional encodings.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention (NO positional encoding!)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, set_size, input_dim]
        Returns:
            encoded: [batch, set_size, hidden_dim]
        """
        # Embed inputs
        x = self.embedding(x)
        
        # Self-attention (order-invariant!)
        attn_out, _ = self.self_attention(x, x, x)
        
        # Residual + layer norm
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.ff(x)
        
        # Residual + layer norm
        x = self.norm2(x + ff_out)
        
        return x


def test_order_invariance():
    """Test that encoder is truly order-invariant."""
    print("🧪 Testing Order Invariance")
    print("=" * 60)
    
    batch_size = 2
    set_size = 5
    input_dim = 2
    hidden_dim = 32
    
    encoder = SetEncoder(input_dim, hidden_dim, num_heads=4)
    encoder.eval()  # Disable dropout
    
    # Test 1: Permutation invariance
    print("\n✅ Test 1: Permutation invariance")
    
    points = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0]]
    ])
    
    # Permute first example
    indices = torch.tensor([2, 0, 4, 1, 3])
    points_permuted = points.clone()
    points_permuted[0] = points[0, indices]
    
    with torch.no_grad():
        encoded1 = encoder(points)
        encoded2 = encoder(points_permuted)
    
    # Aggregate (mean pooling is permutation-invariant)
    agg1 = encoded1.mean(dim=1)
    agg2 = encoded2.mean(dim=1)
    
    diff_agg = (agg1[0] - agg2[0]).abs().mean()
    print(f"   Difference in aggregated representation: {diff_agg.item():.6f}")
    print(f"   {'✅ PASS' if diff_agg < 1e-5 else '❌ FAIL'}: Should be nearly identical")
    
    assert diff_agg < 1e-5, "Aggregated representations should be identical"
    
    # Test 2: Why positional encoding breaks this
    print("\n✅ Test 2: Positional encoding breaks invariance")
    
    # Simulate adding position
    pos_encoding = torch.arange(set_size).unsqueeze(0).unsqueeze(2).float() * 0.1
    points_with_pos1 = points + pos_encoding
    points_with_pos2 = points_permuted + pos_encoding
    
    with torch.no_grad():
        encoded_pos1 = encoder(points_with_pos1)
        encoded_pos2 = encoder(points_with_pos2)
    
    agg_pos1 = encoded_pos1.mean(dim=1)
    agg_pos2 = encoded_pos2.mean(dim=1)
    diff_pos = (agg_pos1[0] - agg_pos2[0]).abs().mean()
    
    print(f"   With positional encoding: {diff_pos.item():.6f}")
    print(f"   {'✅ CORRECT' if diff_pos > 0.01 else '❌ WRONG'}: Should be DIFFERENT")
    print("   This proves why we DON'T use positional encoding for sets!")
    
    assert diff_pos > 0.01, "With positional encoding, should be different"
    
    # Test 3: Set operations are permutation-invariant
    print("\n✅ Test 3: Set aggregation operations")
    
    with torch.no_grad():
        encoded = encoder(points[0:1])
    
    # These should all be permutation-invariant
    mean_pool = encoded.mean(dim=1)
    max_pool = encoded.max(dim=1)[0]
    sum_pool = encoded.sum(dim=1)
    
    print(f"   Mean pooling shape: {mean_pool.shape}")
    print(f"   Max pooling shape: {max_pool.shape}")
    print(f"   Sum pooling shape: {sum_pool.shape}")
    print(f"   All are [batch, hidden_dim] - good for set representation!")
    
    print("\n" + "=" * 60)


def visualize_attention():
    """Visualize self-attention pattern."""
    print("\n📊 Visualizing Self-Attention")
    print("=" * 60)
    
    set_size = 4
    input_dim = 2
    hidden_dim = 16
    
    encoder = SetEncoder(input_dim, hidden_dim, num_heads=1)
    encoder.eval()
    
    # 4 corners of a square
    points = torch.tensor([[
        [0.0, 0.0],  # bottom-left
        [1.0, 0.0],  # bottom-right  
        [1.0, 1.0],  # top-right
        [0.0, 1.0],  # top-left
    ]])
    
    print("\nInput: 4 corners of a unit square")
    for i, p in enumerate(points[0]):
        print(f"   Point {i}: ({p[0].item():.1f}, {p[1].item():.1f})")
    
    with torch.no_grad():
        encoded = encoder(points)
    
    print(f"\nEncoded shape: {encoded.shape}")
    print("\n💡 Key insight:")
    print("   Each point's encoding depends on ALL points")
    print("   But NOT on their order!")
    print("   This is the magic of self-attention WITHOUT positional encoding")


if __name__ == "__main__":
    test_order_invariance()
    visualize_attention()
    
    print("\n" + "=" * 60)
    print("🎯 Solution 2 Summary")
    print("=" * 60)
    print("""
Key implementation details:

1. **NO Positional Encoding**
   - Standard Transformers add sin/cos position embeddings
   - We deliberately SKIP this step
   - This makes encoder permutation-invariant

2. **Self-Attention Mechanism**
   - Each element attends to ALL other elements
   - Creates rich, context-aware representations
   - MultiheadAttention handles the complexity

3. **Residual Connections**
   - x = norm(x + attention(x))
   - Helps gradient flow
   - Preserves original information

4. **Aggregation for Set Representation**
   - Mean pooling: Average all encodings
   - Max pooling: Take maximum across dimension
   - Sum pooling: Sum all encodings
   All are permutation-invariant!

5. **Testing Invariance**
   - Create two orderings of same set
   - Encode both
   - Aggregate and compare
   - Should be (nearly) identical

Why this matters:
- Sets don't have inherent order
- Standard RNNs/Transformers assume order
- This encoder truly treats input as a SET
- Critical for problems like: sorting, TSP, convex hull
    """)
