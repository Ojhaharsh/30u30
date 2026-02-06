"""
Solution 3: Attention Decoder

Reference implementation of a Seq2Seq decoder with Bahdanau attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import attention from solution_1
from solution_1 import AdditiveAttention


class AttentionDecoder(nn.Module):
    """
    GRU Decoder with Bahdanau Attention.
    
    At each step:
    1. Compute attention over encoder outputs using previous hidden state
    2. Get context vector (weighted sum of encoder outputs)
    3. Combine context with current input embedding
    4. Run through GRU to get new hidden state
    5. Predict next token
    
    Key insight: The attention is computed BEFORE the GRU step,
    using the previous hidden state. This is the original Bahdanau design.
    (Some variants compute attention after the GRU step instead.)
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_hidden_size=None,
                 num_layers=1, dropout=0.1, pad_idx=0):
        """
        Args:
            vocab_size: Size of output vocabulary
            embed_size: Dimension of embeddings
            hidden_size: Hidden size of decoder GRU
            encoder_hidden_size: Size of encoder outputs (default: same as hidden_size)
            num_layers: Number of GRU layers
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        
        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # Attention mechanism
        self.attention = AdditiveAttention(encoder_hidden_size, hidden_size)
        
        # GRU input: embedding + context vector
        self.gru = nn.GRU(
            input_size=embed_size + encoder_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: hidden + context + embedding -> vocabulary
        self.output_layer = nn.Linear(
            hidden_size + encoder_hidden_size + embed_size,
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward_step(self, input_token, hidden, encoder_outputs, src_mask=None):
        """
        One decoding step.
        
        Args:
            input_token: Current input token (batch,) - just the token IDs
            hidden: Previous hidden state (num_layers, batch, hidden_size)
            encoder_outputs: Encoder outputs (batch, src_len, encoder_hidden_size)
            src_mask: Source mask where True = ignore (batch, src_len)
            
        Returns:
            output: Vocabulary logits (batch, vocab_size)
            hidden: New hidden state (num_layers, batch, hidden_size)
            attention_weights: Attention distribution (batch, src_len)
        """
        batch_size = input_token.size(0)
        
        # Embed input token: (batch, embed_size)
        embedded = self.dropout(self.embedding(input_token))
        
        # Compute attention using top layer of hidden state
        # hidden: (num_layers, batch, hidden_size)
        # We use the last layer for attention: (batch, hidden_size)
        query = hidden[-1]
        
        context, attention_weights = self.attention(query, encoder_outputs, src_mask)
        
        # Combine embedding and context for GRU input
        # (batch, embed_size + encoder_hidden_size)
        gru_input = torch.cat([embedded, context], dim=1)
        gru_input = gru_input.unsqueeze(1)  # (batch, 1, embed_size + context_size)
        
        # Run GRU
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)  # (batch, hidden_size)
        
        # Compute vocabulary distribution
        # Combine hidden state, context, and embedding (as in paper)
        combined = torch.cat([output, context, embedded], dim=1)
        output = self.output_layer(combined)  # (batch, vocab_size)
        
        return output, hidden, attention_weights
    
    def forward(self, trg, encoder_hidden, encoder_outputs, src_mask=None, 
                teacher_forcing_ratio=1.0):
        """
        Full decoding pass (training mode with teacher forcing).
        
        Args:
            trg: Target sequence (batch, trg_len) - includes SOS but we don't predict it
            encoder_hidden: Initial hidden state from encoder
            encoder_outputs: Encoder outputs for attention
            src_mask: Source padding mask
            teacher_forcing_ratio: Probability of using ground truth as next input
            
        Returns:
            outputs: Vocabulary logits (batch, trg_len-1, vocab_size)
            attentions: Attention weights (batch, trg_len-1, src_len)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.output_layer.out_features
        src_len = encoder_outputs.size(1)
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, trg_len - 1, vocab_size, device=trg.device)
        attentions = torch.zeros(batch_size, trg_len - 1, src_len, device=trg.device)
        
        # Initial hidden state from encoder
        hidden = encoder_hidden
        
        # First input is SOS token (position 0)
        input_token = trg[:, 0]
        
        for t in range(trg_len - 1):
            output, hidden, attn = self.forward_step(
                input_token, hidden, encoder_outputs, src_mask
            )
            
            outputs[:, t] = output
            attentions[:, t] = attn
            
            # Teacher forcing: use ground truth or prediction?
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = trg[:, t + 1]  # Ground truth
            else:
                input_token = output.argmax(dim=-1)  # Prediction
        
        return outputs, attentions


# ============================================================================
# Testing
# ============================================================================

def test_attention_decoder():
    """Comprehensive test of AttentionDecoder."""
    print("Testing AttentionDecoder...")
    
    vocab_size = 100
    embed_size = 64
    hidden_size = 128
    encoder_hidden_size = 128
    batch_size = 3
    src_len = 7
    trg_len = 5
    
    decoder = AttentionDecoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size,
        num_layers=2,
        dropout=0.1
    )
    
    # Create test inputs
    trg = torch.randint(3, vocab_size, (batch_size, trg_len))
    trg[:, 0] = 1  # SOS token
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_size)
    encoder_hidden = torch.randn(2, batch_size, hidden_size)
    
    # Test forward
    outputs, attentions = decoder(trg, encoder_hidden, encoder_outputs)
    
    assert outputs.shape == (batch_size, trg_len - 1, vocab_size), \
        f"Wrong output shape: {outputs.shape}"
    assert attentions.shape == (batch_size, trg_len - 1, src_len), \
        f"Wrong attention shape: {attentions.shape}"
    
    print("  Output shapes correct")
    
    # Check attention sums to 1
    attn_sums = attentions.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
        "Attention weights don't sum to 1"
    
    print("  Attention weights sum to 1")
    
    # Test forward_step
    input_token = torch.randint(3, vocab_size, (batch_size,))
    hidden = torch.randn(2, batch_size, hidden_size)
    
    output, new_hidden, attn = decoder.forward_step(
        input_token, hidden, encoder_outputs
    )
    
    assert output.shape == (batch_size, vocab_size), \
        f"Wrong step output shape: {output.shape}"
    assert new_hidden.shape == hidden.shape, \
        f"Wrong step hidden shape: {new_hidden.shape}"
    assert attn.shape == (batch_size, src_len), \
        f"Wrong step attention shape: {attn.shape}"
    
    print("  Forward step shapes correct")
    
    # Test gradient flow
    outputs.sum().backward()
    assert decoder.embedding.weight.grad is not None, "No gradient for embeddings!"
    assert decoder.attention.W_h.weight.grad is not None, "No gradient for attention!"
    
    print("  Gradients flow correctly")
    
    # Test with mask
    decoder.zero_grad()
    src_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    src_mask[:, -2:] = True  # Mask last 2 positions
    
    outputs_masked, attentions_masked = decoder(
        trg, encoder_hidden, encoder_outputs, src_mask
    )
    
    # Masked positions should have ~0 attention
    masked_attn = attentions_masked[:, :, -2:]
    assert (masked_attn < 1e-5).all(), "Masked positions have attention weight!"
    
    print("  Masking works correctly")
    
    print("Test passed.")
    return decoder


if __name__ == '__main__':
    test_attention_decoder()
