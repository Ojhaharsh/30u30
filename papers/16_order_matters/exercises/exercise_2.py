"""
Exercise 2: Order-Invariant Set Encoder

Learn how to encode sets where order doesn't matter.

Key challenge: encoder({a,b,c}) should equal encoder({c,a,b})

Real-world analogy: A bag of groceries - it doesn't matter if you packed
                   milk then eggs, or eggs then milk. The bag contains
                   the same items!

Your task: Build an encoder that's truly order-invariant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """
    Order-invariant encoder using self-attention WITHOUT positional encodings.
    
    The key insight: Self-attention is permutation-invariant IF we don't
    add positional encodings!
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # TODO: Embedding layer to project inputs to hidden_dim
        self.embedding = None  # TODO: nn.Linear(input_dim, hidden_dim)
        
        # TODO: Multi-head self-attention
        # CRITICAL: We do NOT add positional encodings!
        self.self_attention = None  # TODO: nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # TODO: Feedforward network
        self.ff = None  # TODO: nn.Sequential with 2 Linear layers + ReLU
        # Hint: hidden_dim -> hidden_dim*4 -> hidden_dim
        
        # TODO: Layer normalization
        self.norm1 = None  # TODO: nn.LayerNorm(hidden_dim)
        self.norm2 = None  # TODO: nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Encode a set in an order-invariant way.
        
        Args:
            x: [batch, set_size, input_dim]
            
        Returns:
            encoded: [batch, set_size, hidden_dim]
        """
        # TODO: Step 1 - Embed inputs
        x = None  # TODO: Apply self.embedding
        
        # TODO: Step 2 - Self-attention (order-invariant!)
        # Hint: self.self_attention(x, x, x) returns (output, attention_weights)
        attn_out, _ = None, None  # TODO: Apply self-attention
        
        # TODO: Step 3 - Residual connection + layer norm
        x = None  # TODO: self.norm1(x + attn_out)
        
        # TODO: Step 4 - Feedforward
        ff_out = None  # TODO: Apply self.ff
        
        # TODO: Step 5 - Residual connection + layer norm
        x = None  # TODO: self.norm2(x + ff_out)
        
        return x


def test_order_invariance():
    """
    Test that the encoder is truly order-invariant!
    
    This is the key property: Different orderings of the same set
    should produce the same encoded representation.
    """
    print("🧪 Testing Order Invariance")
    print("=" * 60)
    
    # Setup
    batch_size = 2
    set_size = 5
    input_dim = 2
    hidden_dim = 32
    
    # Create encoder
    encoder = SetEncoder(input_dim, hidden_dim, num_heads=4)
    encoder.eval()  # Disable dropout for deterministic test
    
    # Test 1: Same elements, different orders
    print("\n✅ Test 1: Permutation invariance")
    
    # Create a set of points
    points = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0]]
    ])
    
    # Permute the second batch (shuffle the order)
    indices = torch.tensor([2, 0, 4, 1, 3])  # Random permutation
    points_permuted = points.clone()
    points_permuted[0] = points[0, indices]
    
    with torch.no_grad():
        encoded1 = encoder(points)
        encoded2 = encoder(points_permuted)
    
    # Check if encodings match after sorting
    # We need to sort because self-attention outputs might be in different orders
    # But the SET of encodings should be the same
    encoded1_sorted = torch.sort(encoded1[0].sum(dim=1))[0]
    encoded2_sorted = torch.sort(encoded2[0].sum(dim=1))[0]
    
    diff = (encoded1_sorted - encoded2_sorted).abs().mean()
    print(f"   Difference after permutation: {diff.item():.6f}")
    print(f"   {'✅ PASS' if diff < 0.01 else '❌ FAIL'}: Encodings should be similar")
    
    # Test 2: Aggregation is permutation-invariant
    print("\n✅ Test 2: Aggregated representation")
    
    # Mean pooling is permutation-invariant
    agg1 = encoded1.mean(dim=1)
    agg2 = encoded2.mean(dim=1)
    
    diff_agg = (agg1[0] - agg2[0]).abs().mean()
    print(f"   Difference in aggregated representation: {diff_agg.item():.6f}")
    print(f"   {'✅ PASS' if diff_agg < 1e-5 else '❌ FAIL'}: Should be identical")
    
    # Test 3: Compare with position-aware encoder
    print("\n✅ Test 3: Why positional encoding breaks order invariance")
    print("   (This is a conceptual test - showing what NOT to do)")
    
    # If we added positional encodings, encodings would be different
    # Let's simulate by adding position
    pos_encoding = torch.arange(set_size).unsqueeze(0).unsqueeze(2).float()
    points_with_pos1 = points + pos_encoding * 0.1
    points_with_pos2 = points_permuted + pos_encoding * 0.1
    
    with torch.no_grad():
        encoded_pos1 = encoder(points_with_pos1)
        encoded_pos2 = encoder(points_with_pos2)
    
    agg_pos1 = encoded_pos1.mean(dim=1)
    agg_pos2 = encoded_pos2.mean(dim=1)
    diff_pos = (agg_pos1[0] - agg_pos2[0]).abs().mean()
    
    print(f"   With positional encoding: {diff_pos.item():.6f}")
    print(f"   {'✅ CORRECT' if diff_pos > 0.01 else '❌ WRONG'}: Should be DIFFERENT")
    print("   This shows why we DON'T use positional encoding for sets!")
    
    print("\n" + "=" * 60)


def visualize_attention():
    """Visualize how self-attention treats all elements equally."""
    print("\n📊 Visualizing Self-Attention")
    print("=" * 60)
    
    set_size = 4
    input_dim = 2
    hidden_dim = 16
    
    encoder = SetEncoder(input_dim, hidden_dim, num_heads=1)
    encoder.eval()
    
    # Simple set: 4 corners of a square
    points = torch.tensor([[
        [0.0, 0.0],  # bottom-left
        [1.0, 0.0],  # bottom-right  
        [1.0, 1.0],  # top-right
        [0.0, 1.0],  # top-left
    ]])
    
    print("\nInput points (corners of a square):")
    for i, p in enumerate(points[0]):
        print(f"   Point {i}: {p.tolist()}")
    
    with torch.no_grad():
        encoded = encoder(points)
    
    print(f"\nEncoded shape: {encoded.shape}")
    print("\n💡 Key insight: Each point attends to ALL other points equally")
    print("   (because the set has no inherent order)")


if __name__ == "__main__":
    test_order_invariance()
    visualize_attention()
    
    print("\n" + "=" * 60)
    print("🎯 Exercise 2 Summary")
    print("=" * 60)
    print("""
You've built an order-invariant set encoder!

Key concepts you learned:
1. ✅ Self-attention WITHOUT positional encoding = order invariance
2. ✅ Permutation invariance: encoder(shuffle(x)) = shuffle(encoder(x))
3. ✅ Aggregation (mean/max pooling) creates a single set representation
4. ✅ Why Transformers use positional encoding (for sequences, not sets)

Next: Exercise 3 - Train the model to sort numbers!
    """)
