"""
Exercise 2: Build the Bidirectional Encoder

The encoder's job is to read the entire input sequence and produce
a representation at each position that captures the FULL context
(both what came before AND what comes after).

Why bidirectional?
- "The bank was steep" → 'bank' means riverbank (context: steep)
- "I went to the bank" → 'bank' means financial (context: went)

Without seeing the FULL sentence, we can't know which meaning!

Your task: Implement a bidirectional GRU encoder.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionalEncoder(nn.Module):
    """
    TODO: Implement a bidirectional GRU encoder.
    
    Architecture:
    - Embedding layer
    - Bidirectional GRU
    - Output: concatenated forward and backward hidden states
    
    Args:
        vocab_size: Size of vocabulary
        embed_size: Dimension of embeddings
        hidden_size: Dimension of GRU hidden states (each direction)
        num_layers: Number of GRU layers
        dropout: Dropout probability
        padding_idx: Index of padding token (for embedding)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # TODO: Create the layers
        # 1. Embedding layer with padding_idx
        # 2. Bidirectional GRU with dropout (if num_layers > 1)
        # 3. Dropout layer
        
        self.embedding = None  # TODO
        self.gru = None        # TODO
        self.dropout = None    # TODO
        
        raise NotImplementedError("Implement __init__!")
    
    def forward(
        self, 
        src: torch.Tensor,        # [batch, src_len]
        src_lengths: torch.Tensor  # [batch]
    ):
        """
        Encode the source sequence.
        
        TODO:
        1. Embed the source tokens
        2. Apply dropout
        3. Pack the sequence (for efficiency with padding)
        4. Pass through GRU
        5. Unpack the output
        6. Process hidden state for decoder initialization
        
        Returns:
            outputs: [batch, src_len, hidden_size * 2]
                     (concatenated forward + backward)
            hidden: [num_layers, batch, hidden_size]
                    (combined for decoder initialization)
        """
        raise NotImplementedError("Implement forward!")
    
    def _combine_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Combine forward and backward hidden states for decoder.
        
        The GRU returns hidden as [num_layers * 2, batch, hidden].
        We need to combine the two directions into [num_layers, batch, hidden].
        
        Common strategies:
        1. Sum: h_combined = h_forward + h_backward
        2. Average: h_combined = (h_forward + h_backward) / 2
        3. Concatenate + project: h_combined = W * [h_forward; h_backward]
        
        TODO: Implement option 1 (sum) for simplicity.
        
        Hint:
        - Reshape to [num_layers, 2, batch, hidden]
        - Sum along the direction dimension
        """
        raise NotImplementedError("Implement _combine_hidden!")


# ============================================================================
# Tests
# ============================================================================

def test_encoder():
    """Test your encoder implementation."""
    print("Testing Bidirectional Encoder...")
    
    # Parameters
    batch_size = 4
    src_len = 10
    vocab_size = 100
    embed_size = 64
    hidden_size = 128
    num_layers = 2
    
    # Create encoder
    encoder = BidirectionalEncoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Create dummy input
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    # Add some padding
    src[0, 7:] = 0
    src[1, 8:] = 0
    src_lengths = torch.tensor([7, 8, 10, 10])
    
    # Forward pass
    outputs, hidden = encoder(src, src_lengths)
    
    # Check output shape: [batch, src_len, hidden * 2] (bidirectional)
    expected_output_shape = (batch_size, src_len, hidden_size * 2)
    assert outputs.shape == expected_output_shape, \
        f"Output shape wrong: {outputs.shape}, expected {expected_output_shape}"
    
    # Check hidden shape: [num_layers, batch, hidden]
    expected_hidden_shape = (num_layers, batch_size, hidden_size)
    assert hidden.shape == expected_hidden_shape, \
        f"Hidden shape wrong: {hidden.shape}, expected {expected_hidden_shape}"
    
    # Check that padding positions have some output (they get backward context)
    print(f"   Output norm at padded position: {outputs[0, 8].norm().item():.4f}")
    
    print("✅ All tests passed!")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Hidden shape: {hidden.shape}")


if __name__ == '__main__':
    test_encoder()
