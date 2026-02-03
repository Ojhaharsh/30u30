"""
Exercise 3: Build the Attention Decoder

The decoder is where the magic happens! At each step, it:
1. Looks at what it generated so far (previous token)
2. Asks "what should I focus on?" (attention over encoder)
3. Combines this with its memory (hidden state)
4. Generates the next token

Think of it like writing a summary:
- You keep re-reading the relevant parts of the original text
- Each sentence you write requires looking at different parts
- The attention mechanism is your "eyes" scanning the source

Your task: Implement the attention decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionDecoder(nn.Module):
    """
    TODO: Implement a GRU decoder with Bahdanau attention.
    
    At each step:
    1. Embed the previous token
    2. Compute attention over encoder outputs
    3. Concatenate embedding with attention context
    4. Pass through GRU
    5. Combine outputs for prediction
    
    Args:
        vocab_size: Target vocabulary size
        embed_size: Embedding dimension
        hidden_size: Decoder GRU hidden size
        encoder_hidden_size: Encoder hidden size (will be doubled for bidirectional)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        encoder_hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Encoder is bidirectional
        self.encoder_output_size = encoder_hidden_size * 2
        
        # TODO: Create the layers
        # 1. Embedding layer
        # 2. Attention mechanism (use the one from exercise_1 or import from solutions)
        # 3. GRU layer (input = embedding + context)
        # 4. Output layer (input = hidden + context + embedding)
        # 5. Dropout layer
        
        # Hint: GRU input size = embed_size + encoder_output_size
        # Hint: Output layer input = hidden_size + encoder_output_size + embed_size
        
        raise NotImplementedError("Implement __init__!")
    
    def forward_step(
        self,
        prev_token: torch.Tensor,     # [batch]
        hidden: torch.Tensor,          # [1, batch, hidden_size]
        encoder_outputs: torch.Tensor, # [batch, src_len, encoder_output_size]
        src_mask: torch.Tensor = None  # [batch, src_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single decoding step.
        
        TODO:
        1. Embed prev_token
        2. Compute attention using hidden[-1] as query
        3. Concatenate [embedding; context] for GRU input
        4. Run one GRU step
        5. Combine [hidden; context; embedding] for output
        6. Project to vocabulary
        
        Returns:
            output: [batch, vocab_size] - logits over vocabulary
            hidden: Updated hidden state
            attn_weights: [batch, src_len] - attention weights
        """
        raise NotImplementedError("Implement forward_step!")
    
    def forward(
        self,
        trg: torch.Tensor,             # [batch, trg_len]
        hidden: torch.Tensor,          # Initial hidden from encoder
        encoder_outputs: torch.Tensor, # [batch, src_len, enc_hidden*2]
        src_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence decoding with teacher forcing.
        
        Teacher forcing means we use the GROUND TRUTH previous token,
        not our own predictions. This makes training more stable.
        
        TODO:
        1. Loop through target positions (skip last one)
        2. Call forward_step at each position
        3. Collect outputs and attention weights
        4. Stack into tensors
        
        Returns:
            outputs: [batch, trg_len-1, vocab_size]
            attentions: [batch, trg_len-1, src_len]
        """
        raise NotImplementedError("Implement forward!")


# ============================================================================
# Simplified Attention for Testing (copy from exercise_1 solution if needed)
# ============================================================================

class SimpleAttention(nn.Module):
    """A working attention module for testing the decoder."""
    
    def __init__(self, hidden_size, query_size, key_size):
        super().__init__()
        self.query_proj = nn.Linear(query_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(key_size, hidden_size, bias=False)
        self.energy = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, query, keys, mask=None):
        query_proj = self.query_proj(query).unsqueeze(1)
        keys_proj = self.key_proj(keys)
        scores = self.energy(torch.tanh(query_proj + keys_proj)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


# ============================================================================
# Tests
# ============================================================================

def test_decoder():
    """Test your decoder implementation."""
    print("Testing Attention Decoder...")
    
    # Parameters
    batch_size = 4
    src_len = 10
    trg_len = 8
    vocab_size = 100
    embed_size = 64
    hidden_size = 128
    encoder_hidden_size = 128  # Will be doubled for bidirectional
    
    # Create decoder
    decoder = AttentionDecoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size
    )
    
    # Dummy inputs
    trg = torch.randint(1, vocab_size, (batch_size, trg_len))
    hidden = torch.randn(1, batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_size * 2)
    
    # Test single step
    prev_token = trg[:, 0]
    output, new_hidden, attn = decoder.forward_step(
        prev_token, hidden, encoder_outputs
    )
    
    assert output.shape == (batch_size, vocab_size), \
        f"Single step output shape wrong: {output.shape}"
    assert new_hidden.shape == hidden.shape, \
        f"Hidden shape changed: {new_hidden.shape}"
    assert attn.shape == (batch_size, src_len), \
        f"Attention shape wrong: {attn.shape}"
    
    # Test full sequence
    outputs, attentions = decoder(trg, hidden, encoder_outputs)
    
    assert outputs.shape == (batch_size, trg_len - 1, vocab_size), \
        f"Full output shape wrong: {outputs.shape}"
    assert attentions.shape == (batch_size, trg_len - 1, src_len), \
        f"Full attention shape wrong: {attentions.shape}"
    
    # Check attention sums to 1
    attn_sums = attentions[0, 0].sum()
    assert abs(attn_sums - 1.0) < 1e-5, \
        f"Attention doesn't sum to 1: {attn_sums}"
    
    print("✅ All tests passed!")
    print(f"   Single step output: {output.shape}")
    print(f"   Full output: {outputs.shape}")
    print(f"   Attention weights: {attentions.shape}")


if __name__ == '__main__':
    test_decoder()
