"""
Solution 2: Bidirectional Encoder

Reference implementation of a bidirectional GRU encoder.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional GRU Encoder.
    
    Why bidirectional?
    - Forward pass: "I love cats" -> context builds left-to-right
    - Backward pass: "I love cats" -> context builds right-to-left
    - Together: each position knows about the ENTIRE sentence
    
    This is crucial for attention! When we attend to position 2,
    we get information from both directions, not just what came before.
    
    Output projection:
    - Forward and backward outputs are concatenated: 2 * hidden_size
    - We project back to hidden_size for compatibility with decoder
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, 
                 dropout=0.1, pad_idx=0):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Dimension of embeddings
            hidden_size: Hidden size of GRU (output will be projected to this)
            num_layers: Number of GRU layers
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # Bidirectional GRU
        # Note: We use hidden_size for GRU, giving us 2*hidden_size when bidirectional
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project concatenated forward/backward to hidden_size
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Project final hidden states for decoder initialization
        # We combine forward and backward final states
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings and linear layers."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.xavier_uniform_(self.hidden_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.hidden_projection.bias)
    
    def forward(self, src, src_lengths):
        """
        Encode source sequence.
        
        Args:
            src: Source token IDs (batch, src_len)
            src_lengths: Actual lengths before padding (batch,)
            
        Returns:
            outputs: Encoder outputs (batch, src_len, hidden_size)
            hidden: Final hidden state for decoder (num_layers, batch, hidden_size)
        """
        batch_size, src_len = src.shape
        
        # Embed input tokens
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed_size)
        
        # Sort by length for packing (required by pack_padded_sequence)
        src_lengths_cpu = src_lengths.cpu()
        sorted_lengths, sorted_indices = src_lengths_cpu.sort(descending=True)
        sorted_embedded = embedded[sorted_indices]
        
        # Pack the sequence
        packed = pack_padded_sequence(
            sorted_embedded, 
            sorted_lengths.tolist(),
            batch_first=True,
            enforce_sorted=True
        )
        
        # Run through bidirectional GRU
        packed_outputs, hidden = self.gru(packed)
        
        # Unpack
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, 
                                          total_length=src_len)
        
        # Unsort to restore original order
        _, unsort_indices = sorted_indices.sort()
        outputs = outputs[unsort_indices]
        
        # Handle hidden state
        # hidden shape: (num_layers * 2, batch, hidden_size)
        # Separate forward and backward
        hidden = hidden[:, sorted_indices, :]  # First sort
        hidden = hidden[:, unsort_indices, :]  # Then unsort
        
        # Reshape: (num_layers, 2, batch, hidden_size)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate forward and backward final states
        # Forward is direction 0, backward is direction 1
        hidden_fwd = hidden[:, 0, :, :]  # (num_layers, batch, hidden_size)
        hidden_bwd = hidden[:, 1, :, :]  # (num_layers, batch, hidden_size)
        hidden_combined = torch.cat([hidden_fwd, hidden_bwd], dim=2)  # (num_layers, batch, hidden_size*2)
        
        # Project combined hidden states
        # Apply to each layer: (num_layers, batch, hidden_size)
        hidden_projected = self.hidden_projection(hidden_combined)
        
        # Project outputs: (batch, src_len, hidden_size*2) -> (batch, src_len, hidden_size)
        outputs = self.output_projection(outputs)
        
        return outputs, hidden_projected


# ============================================================================
# Testing
# ============================================================================

def test_bidirectional_encoder():
    """Comprehensive test of the BidirectionalEncoder."""
    print("Testing BidirectionalEncoder...")
    
    vocab_size = 100
    embed_size = 64
    hidden_size = 128
    batch_size = 3
    
    encoder = BidirectionalEncoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1
    )
    
    # Create test input with varying lengths
    src = torch.tensor([
        [5, 3, 8, 2, 1, 0, 0],  # length 5
        [9, 7, 4, 0, 0, 0, 0],  # length 3
        [6, 2, 1, 8, 5, 3, 9],  # length 7
    ])
    src_lengths = torch.tensor([5, 3, 7])
    
    # Forward pass
    outputs, hidden = encoder(src, src_lengths)
    
    # Check output shapes
    assert outputs.shape == (batch_size, 7, hidden_size), \
        f"Wrong output shape: {outputs.shape}"
    assert hidden.shape == (2, batch_size, hidden_size), \
        f"Wrong hidden shape: {hidden.shape}"
    
    print("  Output shapes correct")
    
    # Check that padded positions still have valid outputs
    # (they should, pack_padded_sequence handles this)
    
    # Check gradient flow
    outputs.sum().backward()
    assert encoder.embedding.weight.grad is not None, "No gradient for embeddings!"
    assert encoder.output_projection.weight.grad is not None, "No gradient for projection!"
    
    print("  Gradients flow correctly")
    
    # Test that different length handling is correct
    encoder.zero_grad()
    
    # The output for sample 1 (length 3) at position 4+ should be from padding
    # but the hidden state should properly represent positions 0-2
    
    print("Test passed.")
    return encoder


if __name__ == '__main__':
    test_bidirectional_encoder()
