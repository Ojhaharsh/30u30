"""
Pointer Networks and Set Processing Implementation
===================================================

This module implements the core components for processing sets with neural networks:
1. Pointer Networks - for selecting elements from input
2. Set Encoders - for order-invariant encoding
3. Read-Process-Write framework

Paper: "Order Matters: Sequence to Sequence for Sets"
       Vinyals et al. (2015)
       https://arxiv.org/abs/1511.06391
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointerNetwork(nn.Module):
    """
    Pointer Network for selecting elements from a set.
    
    Instead of generating tokens from a vocabulary, this network "points"
    to elements in the input sequence.
    
    Key property: The output space is the indices of the input set.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: Process the input set
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: Generate pointers
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # Attention mechanism for pointing
        # Formula: e_i = v^T tanh(W1*encoder_i + W2*decoder)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        teacher_forcing: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Forward pass for pointer network.
        
        Args:
            inputs: [batch, seq_len, input_dim] - The input set (possibly padded)
            input_lengths: [batch] - Actual lengths before padding
            max_steps: Number of elements to point to
            teacher_forcing: [batch, max_steps] - Ground truth pointers
            temperature: Softmax temperature
            
        Returns:
            pointers: [batch, max_steps] - Selected indices
            log_probs: [batch, max_steps] - Log probabilities
            attention_weights: List of [batch, seq_len] for visualization
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        device = inputs.device
        
        # Phase 1: Encode
        encoder_outputs, (hidden, cell) = self.encoder(inputs)
        
        # Initialize decoder state
        decoder_hidden = hidden[-1]
        decoder_cell = cell[-1]
        
        # Start with a zero vector (no previous selection)
        decoder_input = torch.zeros(batch_size, self.input_dim, device=device)
        
        # Padding mask: 1 = valid, 0 = padding
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < input_lengths.unsqueeze(1)).float()
        
        # Phase 2: Decode
        pointers = []
        log_probs_list = []
        attention_weights_list = []
        
        # Track selected elements to enable sampling without replacement
        selection_mask = mask.clone()
        
        for step in range(max_steps):
            decoder_hidden, decoder_cell = self.decoder_cell(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            decoder_hidden = self.dropout(decoder_hidden)
            
            # Attention Mechanism (Eq 1 in Paper)
            encoder_proj = self.W1(encoder_outputs)
            decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
            combined = torch.tanh(encoder_proj + decoder_proj)
            scores = self.v(combined).squeeze(-1) / temperature
            
            # Mask padding and selected elements
            scores = scores.masked_fill(selection_mask == 0, float('-inf'))
            
            probs = F.softmax(scores, dim=-1)
            log_probs = F.log_softmax(scores, dim=-1)
            
            if teacher_forcing is not None and step < teacher_forcing.size(1):
                pointer = teacher_forcing[:, step]
            else:
                pointer = torch.argmax(probs, dim=-1)
            
            pointers.append(pointer)
            log_probs_list.append(log_probs.gather(1, pointer.unsqueeze(1)))
            attention_weights_list.append(probs)
            
            selection_mask = selection_mask.scatter(1, pointer.unsqueeze(1), 0)
            decoder_input = inputs[torch.arange(batch_size, device=device), pointer]
        
        pointers = torch.stack(pointers, dim=1)
        log_probs = torch.cat(log_probs_list, dim=1)
        
        return pointers, log_probs, attention_weights_list


class SetEncoder(nn.Module):
    """
    Order-invariant encoder for sets.
    Ensures encoder({a, b}) = encoder({b, a}).
    Uses self-attention without positional encodings.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed inputs
        x = self.embedding(x)
        
        # Self-attention without positional encodings creates order invariance
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class ReadProcessWrite(nn.Module):
    """
    Implementation of the Read-Process-Write framework.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_set_encoder: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.use_set_encoder = use_set_encoder
        
        if use_set_encoder:
            # Phase 1: READ
            self.encoder = SetEncoder(input_dim, hidden_dim, **kwargs)
            # Phase 3: WRITE
            self.pointer_net = PointerNetwork(hidden_dim, hidden_dim, **kwargs)
        else:
            self.pointer_net = PointerNetwork(input_dim, hidden_dim, **kwargs)
    
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        if self.use_set_encoder:
            encoded = self.encoder(inputs)
            pointers, log_probs, attentions = self.pointer_net(
                encoded, input_lengths, max_steps, **kwargs
            )
        else:
            pointers, log_probs, attentions = self.pointer_net(
                inputs, input_lengths, max_steps, **kwargs
            )
        
        return pointers, log_probs, attentions


def compute_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_log_probs = log_probs.gather(1, targets)
    return -target_log_probs.mean()


def accuracy(pointers: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (pointers == targets).all(dim=1).float()
    return correct.mean().item()


if __name__ == "__main__":
    print("Pointer Network implementation loaded.")
