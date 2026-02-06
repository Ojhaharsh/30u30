"""
Day 15: Bahdanau Attention - PyTorch Implementation
Based on Dzmitry Bahdanau's "Jointly Learning to Align and Translate" (2014)
Heavily commented for educational purposes

This is a complete implementation of a Seq2Seq model with additive attention.
Every calculation step is explained. No magic.
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
    
    Parameters:
    -----------
    hidden_size : int
        Dimension of the attention hidden layer
    key_size : int
        Dimension of encoder hidden states
    query_size : int
        Dimension of decoder hidden states
    """
    
    def __init__(self, hidden_size: int, key_size: int = None, query_size: int = None):
        super().__init__()
        
        key_size = key_size or hidden_size
        query_size = query_size or hidden_size
        
        # Project query (decoder hidden)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        # Project keys (encoder outputs)
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        # Final energy projection
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Standard Xavier initialization."""
        for layer in [self.query_layer, self.key_layer, self.energy_layer]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(
        self, 
        query: torch.Tensor, 
        keys: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates context vectors and attention weights.
        
        Parameters:
        -----------
        query : torch.Tensor
            Previous decoder hidden state [batch, query_size]
        keys : torch.Tensor
            Encoder hidden states [batch, src_len, key_size]
        mask : torch.Tensor, optional
            Padding mask [batch, src_len]
        """
        batch_size, src_len, _ = keys.size()
        
        # Transform query and keys to hidden space
        query_proj = self.query_layer(query).unsqueeze(1)
        keys_proj = self.key_layer(keys)
        
        # Additive scoring: tanh(W_q*q + W_k*k)
        scores = torch.tanh(query_proj + keys_proj)
        energy = self.energy_layer(scores).squeeze(-1)
        
        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))
        
        weights = F.softmax(energy, dim=-1)
        
        # Context is weighted sum of keys
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        
        return context, weights


class Encoder(nn.Module):
    """
    Bidirectional GRU encoder as described by Bahdanau.
    
    Parameters:
    -----------
    vocab_size : int
        Size of source vocabulary
    embed_size : int
        Word embedding dimension
    hidden_size : int
        GRU hidden dimension
    num_layers : int
        Number of recursive layers
    dropout : float
        Dropout probability
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
        """Encodes source tokens into context-aware hidden states."""
        embedded = self.dropout(self.embedding(src))
        
        packed = pack_padded_sequence(
            embedded, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        # Merge directions for decoder initialization
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, -1, hidden.size(-1)).sum(dim=1)
        
        return outputs, hidden


class AttentionDecoder(nn.Module):
    """
    GRU decoder augmented with Bahdanau attention.
    
    Parameters:
    -----------
    vocab_size : int
        Target vocabulary size
    embed_size : int
        Embedding dimension
    hidden_size : int
        Decoder hidden dimension
    encoder_hidden_size : int
        Internal dimension of encoder states
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int,
        encoder_hidden_size: int,
        num_layers: int = 1, 
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        
        # Attention over 2x hidden (bidirectional) encoder states
        self.attention = BahdanauAttention(
            hidden_size=hidden_size,
            key_size=encoder_hidden_size * 2,
            query_size=hidden_size
        )
        
        self.gru = nn.GRU(
            input_size=embed_size + (encoder_hidden_size * 2),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_size + (encoder_hidden_size * 2) + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward_step(
        self, 
        prev_token: torch.Tensor, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes a single step of the decoding process."""
        embedded = self.dropout(self.embedding(prev_token))
        context, weights = self.attention(hidden[-1], encoder_outputs, src_mask)
        
        gru_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        gru_output, hidden = self.gru(gru_input, hidden)
        
        # Final projection includes embedding and context for stability
        combined = torch.cat([gru_output.squeeze(1), context, embedded], dim=-1)
        logits = self.output_layer(combined)
        
        return logits, hidden, weights


class Seq2SeqWithAttention(nn.Module):
    """
    Complete model combining Encoder and Attention-equipped Decoder.
    """
    
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: torch.Tensor, 
        trg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for teacher-forced training."""
        src_mask = (src == self.pad_idx)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        logits, attentions = self.decoder(trg, hidden, encoder_outputs, src_mask)
        return logits, attentions
    
    @torch.no_grad()
    def translate(
        self, 
        src: torch.Tensor, 
        src_lengths: torch.Tensor,
        max_len: int = 50,
        sos_idx: int = 2,
        eos_idx: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference via greedy decoding."""
        self.eval()
        src_mask = (src == self.pad_idx)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        prev_token = torch.full((src.size(0),), sos_idx, dtype=torch.long, device=src.device)
        translations, attentions = [], []
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        
        for _ in range(max_len):
            logits, hidden, attn = self.decoder.forward_step(prev_token, hidden, encoder_outputs, src_mask)
            prev_token = logits.argmax(dim=-1)
            translations.append(prev_token)
            attentions.append(attn)
            
            finished |= (prev_token == eos_idx)
            if finished.all(): break
            
        return torch.stack(translations, dim=1), torch.stack(attentions, dim=1)


def create_model(src_vocab, trg_vocab, embed_dim=256, hidden_dim=512, layers=2, dropout=0.1, pad=0):
    """Factory helper for model instantiation."""
    enc = Encoder(src_vocab, embed_dim, hidden_dim, layers, dropout, pad)
    dec = AttentionDecoder(trg_vocab, embed_dim, hidden_dim, hidden_dim, layers, dropout, pad)
    model = Seq2SeqWithAttention(enc, dec, pad)
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            
    return model
