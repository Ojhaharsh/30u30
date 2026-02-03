"""
Solution 1: Additive (Bahdanau) Attention

The attention mechanism that changed everything.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style Additive Attention.
    
    The score function: score(s, h) = v^T * tanh(W_s * s + W_h * h)
    
    This is called "additive" because we ADD the transformed query and key,
    unlike dot-product attention which multiplies them.
    
    Why tanh? It squashes values to [-1, 1], acting as a learnable
    similarity function that can capture complex relationships.
    """
    
    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_size=None):
        """
        Args:
            encoder_hidden_size: Size of encoder hidden states (keys/values)
            decoder_hidden_size: Size of decoder hidden state (query)
            attention_size: Size of attention hidden layer (default: decoder_hidden_size)
        """
        super().__init__()
        
        if attention_size is None:
            attention_size = decoder_hidden_size
        
        # Transform encoder hidden states (keys)
        self.W_h = nn.Linear(encoder_hidden_size, attention_size, bias=False)
        
        # Transform decoder hidden state (query)
        self.W_s = nn.Linear(decoder_hidden_size, attention_size, bias=False)
        
        # Final projection to scalar score
        self.v = nn.Linear(attention_size, 1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small random values for stable training."""
        for module in [self.W_h, self.W_s, self.v]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Current decoder state (batch, decoder_hidden_size)
            encoder_outputs: All encoder states (batch, src_len, encoder_hidden_size)
            mask: Boolean mask where True = ignore (batch, src_len)
            
        Returns:
            context: Weighted sum of encoder outputs (batch, encoder_hidden_size)
            attention_weights: Attention distribution (batch, src_len)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # Transform encoder outputs: (batch, src_len, attention_size)
        encoder_proj = self.W_h(encoder_outputs)
        
        # Transform decoder hidden: (batch, attention_size)
        decoder_proj = self.W_s(decoder_hidden)
        
        # Broadcast and add: (batch, src_len, attention_size)
        # decoder_proj is (batch, attention_size), we expand to (batch, 1, attention_size)
        combined = torch.tanh(encoder_proj + decoder_proj.unsqueeze(1))
        
        # Get scalar scores: (batch, src_len)
        scores = self.v(combined).squeeze(-1)
        
        # Apply mask (set masked positions to -inf before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Normalize with softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector: weighted sum of encoder outputs
        # (batch, 1, src_len) @ (batch, src_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, encoder_hidden_size)
        
        return context, attention_weights


# ============================================================================
# Testing
# ============================================================================

def test_additive_attention():
    """Comprehensive test of the AdditiveAttention implementation."""
    print("Testing AdditiveAttention...")
    
    batch_size = 2
    src_len = 5
    encoder_hidden_size = 256
    decoder_hidden_size = 256
    
    # Create attention module
    attention = AdditiveAttention(encoder_hidden_size, decoder_hidden_size)
    
    # Create test inputs
    decoder_hidden = torch.randn(batch_size, decoder_hidden_size)
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_size)
    
    # Test without mask
    context, weights = attention(decoder_hidden, encoder_outputs)
    
    assert context.shape == (batch_size, encoder_hidden_size), \
        f"Wrong context shape: {context.shape}"
    assert weights.shape == (batch_size, src_len), \
        f"Wrong weights shape: {weights.shape}"
    
    # Weights should sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sums}"
    
    print("  ✓ Shape tests passed")
    print("  ✓ Weights sum to 1")
    
    # Test with mask
    mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    mask[:, -2:] = True  # Mask last 2 positions
    
    context_masked, weights_masked = attention(decoder_hidden, encoder_outputs, mask)
    
    # Masked positions should have ~0 weight
    assert (weights_masked[:, -2:] < 1e-5).all(), "Masked positions have non-zero weights!"
    
    # Remaining weights should still sum to 1
    assert torch.allclose(weights_masked.sum(dim=-1), torch.ones(batch_size), atol=1e-5), \
        "Masked weights don't sum to 1"
    
    print("  ✓ Masking works correctly")
    
    # Test gradient flow
    context.sum().backward()
    assert attention.W_h.weight.grad is not None, "No gradients for W_h!"
    assert attention.W_s.weight.grad is not None, "No gradients for W_s!"
    assert attention.v.weight.grad is not None, "No gradients for v!"
    
    print("  ✓ Gradients flow correctly")
    
    print("\n✅ All tests passed!")
    return attention


if __name__ == '__main__':
    test_additive_attention()
