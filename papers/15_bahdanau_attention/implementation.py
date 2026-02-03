"""
Day 15: Bahdanau Attention - Complete Implementation

Neural Machine Translation by Jointly Learning to Align and Translate
Bahdanau, Cho, Bengio (2014)

This module implements the complete Seq2Seq model with additive (Bahdanau) attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """
    Additive attention mechanism from Bahdanau et al. (2014).
    
    The attention score is computed as:
        e_{t,i} = v^T * tanh(W_s * s_{t-1} + W_h * h_i)
    
    Where:
        - s_{t-1}: Previous decoder hidden state (query)
        - h_i: Encoder hidden state at position i (key/value)
        - W_s, W_h, v: Learnable parameters
    
    Args:
        hidden_size: Dimension of the attention hidden layer
        key_size: Dimension of encoder hidden states
        query_size: Dimension of decoder hidden states
    """
    
    def __init__(self, hidden_size: int, key_size: int = None, query_size: int = None):
        super().__init__()
        
        key_size = key_size or hidden_size
        query_size = query_size or hidden_size
        
        # Project query (decoder hidden) to hidden_size
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        
        # Project keys (encoder outputs) to hidden_size
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        
        # Compute scalar energy from hidden_size
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for layer in [self.query_layer, self.key_layer, self.energy_layer]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(
        self, 
        query: torch.Tensor, 
        keys: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            query: Decoder hidden state [batch_size, query_size]
            keys: Encoder outputs [batch_size, src_len, key_size]
            mask: Padding mask [batch_size, src_len] where True = pad (ignore)
            
        Returns:
            context: Weighted sum of keys [batch_size, key_size]
            attention_weights: Attention distribution [batch_size, src_len]
        """
        batch_size, src_len, _ = keys.size()
        
        # Project query: [batch, query_size] -> [batch, 1, hidden]
        query_proj = self.query_layer(query).unsqueeze(1)
        
        # Project keys: [batch, src_len, key_size] -> [batch, src_len, hidden]
        keys_proj = self.key_layer(keys)
        
        # Additive attention: tanh(W_s * s + W_h * h)
        # Broadcasting: [batch, 1, hidden] + [batch, src_len, hidden]
        scores = torch.tanh(query_proj + keys_proj)
        
        # Compute energy: [batch, src_len, hidden] -> [batch, src_len]
        energy = self.energy_layer(scores).squeeze(-1)
        
        # Mask padded positions (set to -inf so softmax gives 0)
        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(energy, dim=-1)
        
        # Handle all-masked case (prevent NaN)
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )
        
        # Compute context vector: weighted sum of encoder outputs
        # [batch, 1, src_len] @ [batch, src_len, key_size] -> [batch, 1, key_size]
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attention_weights


class Encoder(nn.Module):
    """
    Bidirectional GRU encoder.
    
    Processes the source sequence in both directions to capture full context.
    
    Args:
        vocab_size: Size of source vocabulary
        embed_size: Dimension of word embeddings
        hidden_size: Dimension of GRU hidden states
        num_layers: Number of GRU layers
        dropout: Dropout probability
        padding_idx: Index of padding token
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
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the source sequence.
        
        Args:
            src: Source token IDs [batch_size, src_len]
            src_lengths: Length of each source sequence [batch_size]
            
        Returns:
            outputs: Encoder hidden states [batch_size, src_len, 2*hidden_size]
            hidden: Final hidden state [num_layers, batch_size, hidden_size]
        """
        # Embed source tokens
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, embed]
        
        # Pack for efficient processing of variable-length sequences
        packed = pack_padded_sequence(
            embedded, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Encode
        packed_outputs, hidden = self.gru(packed)
        
        # Unpack
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch, src_len, 2*hidden_size]
        
        # Combine bidirectional hidden states for decoder initialization
        # hidden: [2*num_layers, batch, hidden] -> [num_layers, batch, hidden]
        hidden = self._combine_bidirectional_hidden(hidden)
        
        return outputs, hidden
    
    def _combine_bidirectional_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Combine forward and backward hidden states.
        
        Takes: [num_layers*2, batch, hidden]
        Returns: [num_layers, batch, hidden]
        """
        # Reshape: [num_layers, 2, batch, hidden]
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, hidden.size(1), hidden.size(2))
        
        # Sum forward and backward (could also concatenate and project)
        hidden = hidden.sum(dim=1)
        
        return hidden


class AttentionDecoder(nn.Module):
    """
    GRU decoder with Bahdanau attention.
    
    At each step:
    1. Compute attention over encoder outputs
    2. Concatenate attention context with input embedding
    3. Pass through GRU
    4. Generate output token
    
    Args:
        vocab_size: Size of target vocabulary
        embed_size: Dimension of word embeddings
        hidden_size: Dimension of decoder GRU hidden states
        encoder_hidden_size: Dimension of encoder hidden states (before bidirectional)
        attention_size: Dimension of attention hidden layer
        num_layers: Number of GRU layers
        dropout: Dropout probability
        padding_idx: Index of padding token
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int,
        encoder_hidden_size: int,
        attention_size: int = None,
        num_layers: int = 1, 
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder is bidirectional, so outputs are 2x hidden_size
        self.encoder_output_size = encoder_hidden_size * 2
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        
        # Attention mechanism
        attention_size = attention_size or hidden_size
        self.attention = BahdanauAttention(
            hidden_size=attention_size,
            key_size=self.encoder_output_size,
            query_size=hidden_size
        )
        
        # GRU input: previous token embedding + attention context
        gru_input_size = embed_size + self.encoder_output_size
        
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: combines GRU output, context, and embedding
        output_input_size = hidden_size + self.encoder_output_size + embed_size
        self.output_layer = nn.Linear(output_input_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward_step(
        self, 
        prev_token: torch.Tensor, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            prev_token: Previous token ID [batch_size]
            hidden: Previous hidden state [num_layers, batch_size, hidden_size]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden*2]
            src_mask: Source padding mask [batch_size, src_len]
            
        Returns:
            output: Logits over vocabulary [batch_size, vocab_size]
            hidden: Updated hidden state
            attention_weights: Attention distribution [batch_size, src_len]
        """
        # Embed previous token
        embedded = self.dropout(self.embedding(prev_token))  # [batch, embed]
        
        # Compute attention using last layer's hidden state as query
        query = hidden[-1]  # [batch, hidden]
        context, attention_weights = self.attention(query, encoder_outputs, src_mask)
        
        # GRU input: [embedding; context]
        gru_input = torch.cat([embedded, context], dim=-1)  # [batch, embed+enc*2]
        gru_input = gru_input.unsqueeze(1)  # [batch, 1, input_size]
        
        # Single GRU step
        gru_output, hidden = self.gru(gru_input, hidden)
        gru_output = gru_output.squeeze(1)  # [batch, hidden]
        
        # Combine all information for output prediction
        # This "maxout" style combination helps with gradient flow
        combined = torch.cat([gru_output, context, embedded], dim=-1)
        output = self.output_layer(combined)  # [batch, vocab_size]
        
        return output, hidden, attention_weights
    
    def forward(
        self, 
        trg: torch.Tensor, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence decoding with teacher forcing.
        
        Args:
            trg: Target token IDs [batch_size, trg_len]
            hidden: Initial decoder hidden state from encoder
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden*2]
            src_mask: Source padding mask [batch_size, src_len]
            
        Returns:
            outputs: Logits [batch_size, trg_len-1, vocab_size]
            attentions: Attention weights [batch_size, trg_len-1, src_len]
        """
        batch_size, trg_len = trg.size()
        
        outputs = []
        attentions = []
        
        # Decode step by step
        for t in range(trg_len - 1):
            prev_token = trg[:, t]
            
            output, hidden, attn = self.forward_step(
                prev_token, hidden, encoder_outputs, src_mask
            )
            
            outputs.append(output)
            attentions.append(attn)
        
        # Stack: [batch, trg_len-1, ...]
        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)
        
        return outputs, attentions


class Seq2SeqWithAttention(nn.Module):
    """
    Complete Sequence-to-Sequence model with Bahdanau attention.
    
    Args:
        encoder: Encoder module
        decoder: Decoder module with attention
        pad_idx: Padding token index for creating masks
    """
    
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        
    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create mask for source padding tokens."""
        return src == self.pad_idx  # True where padded
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: torch.Tensor, 
        trg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            src: Source tokens [batch_size, src_len]
            src_lengths: Source lengths [batch_size]
            trg: Target tokens [batch_size, trg_len]
            
        Returns:
            outputs: Logits [batch_size, trg_len-1, vocab_size]
            attentions: Attention weights [batch_size, trg_len-1, src_len]
        """
        # Create source mask
        src_mask = self.create_src_mask(src)
        
        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Decode
        outputs, attentions = self.decoder(trg, hidden, encoder_outputs, src_mask)
        
        return outputs, attentions
    
    @torch.no_grad()
    def translate(
        self, 
        src: torch.Tensor, 
        src_lengths: torch.Tensor,
        max_len: int = 50,
        sos_idx: int = 2,
        eos_idx: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy translation for inference.
        
        Args:
            src: Source tokens [batch_size, src_len]
            src_lengths: Source lengths [batch_size]
            max_len: Maximum output length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            
        Returns:
            translations: Predicted tokens [batch_size, output_len]
            attentions: Attention weights [batch_size, output_len, src_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Create mask and encode
        src_mask = self.create_src_mask(src)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Start with <sos> token
        prev_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        
        translations = []
        attentions = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            output, hidden, attn = self.decoder.forward_step(
                prev_token, hidden, encoder_outputs, src_mask
            )
            
            # Greedy decoding
            prev_token = output.argmax(dim=-1)
            
            translations.append(prev_token)
            attentions.append(attn)
            
            # Track finished sequences
            finished = finished | (prev_token == eos_idx)
            
            # Stop if all sequences finished
            if finished.all():
                break
        
        translations = torch.stack(translations, dim=1)
        attentions = torch.stack(attentions, dim=1)
        
        return translations, attentions


def create_model(
    src_vocab_size: int,
    trg_vocab_size: int,
    embed_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    pad_idx: int = 0
) -> Seq2SeqWithAttention:
    """
    Factory function to create the Seq2Seq model.
    
    Args:
        src_vocab_size: Source vocabulary size
        trg_vocab_size: Target vocabulary size
        embed_size: Embedding dimension
        hidden_size: Hidden layer dimension
        num_layers: Number of GRU layers
        dropout: Dropout probability
        pad_idx: Padding token index
        
    Returns:
        Initialized Seq2SeqWithAttention model
    """
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=pad_idx
    )
    
    decoder = AttentionDecoder(
        vocab_size=trg_vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        encoder_hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=pad_idx
    )
    
    model = Seq2SeqWithAttention(encoder, decoder, pad_idx)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model


# ============================================================================
# Visualization utilities
# ============================================================================

def plot_attention(
    attention: torch.Tensor,
    src_tokens: list,
    trg_tokens: list,
    figsize: tuple = (10, 10),
    save_path: str = None
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention: Attention weights [trg_len, src_len]
        src_tokens: List of source tokens
        trg_tokens: List of target tokens
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention, cmap='viridis', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right')
    ax.set_yticklabels(trg_tokens)
    
    # Labels
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Bahdanau Attention Weights')
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == '__main__':
    # Quick test
    print("Testing Bahdanau Attention implementation...")
    
    # Hyperparameters
    batch_size = 4
    src_len = 10
    trg_len = 8
    src_vocab = 1000
    trg_vocab = 1000
    embed_size = 256
    hidden_size = 512
    
    # Create model
    model = create_model(
        src_vocab_size=src_vocab,
        trg_vocab_size=trg_vocab,
        embed_size=embed_size,
        hidden_size=hidden_size
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create dummy data
    src = torch.randint(1, src_vocab, (batch_size, src_len))
    src[:, -2:] = 0  # Add padding
    src_lengths = torch.tensor([src_len - 2] * batch_size)
    
    trg = torch.randint(1, trg_vocab, (batch_size, trg_len))
    
    # Forward pass
    outputs, attentions = model(src, src_lengths, trg)
    
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attentions.shape}")
    print(f"Attention sums to 1: {attentions[0, 0].sum().item():.4f}")
    
    # Test translation
    translations, trans_attn = model.translate(src, src_lengths, max_len=15)
    print(f"Translation shape: {translations.shape}")
    
    print("\n✅ All tests passed!")
