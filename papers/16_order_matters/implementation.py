"""
Pointer Networks and Set Processing Implementation

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
    to elements in the input sequence. Think of it like choosing which
    items to pick from a shelf, rather than naming items from memory.
    
    Key innovation: The output space is the INPUT SET itself!
    
    Example:
        Input: [5, 2, 9, 1]
        Output pointers: [3, 1, 0, 2]  (indices)
        Decoded output: [1, 2, 5, 9]  (sorted!)
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
        # This computes: e_i = v^T tanh(W1*encoder_i + W2*decoder)
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
            inputs: [batch, seq_len, input_dim] - The input SET
            input_lengths: [batch] - Actual lengths before padding
            max_steps: How many elements to point to
            teacher_forcing: [batch, max_steps] - Ground truth pointers (training)
            temperature: Softmax temperature for sampling
            
        Returns:
            pointers: [batch, max_steps] - Selected indices
            log_probs: [batch, max_steps] - Log probabilities
            attention_weights: List of [batch, seq_len] for visualization
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        device = inputs.device
        
        # ===================================================================
        # PHASE 1: ENCODE - Process the input set
        # ===================================================================
        # This creates a rich representation of each element in context
        encoder_outputs, (hidden, cell) = self.encoder(inputs)
        # encoder_outputs: [batch, seq_len, hidden_dim]
        
        # Initialize decoder state from encoder's final state
        # Take the last layer's hidden/cell state
        decoder_hidden = hidden[-1]  # [batch, hidden_dim]
        decoder_cell = cell[-1]      # [batch, hidden_dim]
        
        # Start with a zero vector (no previous selection)
        decoder_input = torch.zeros(batch_size, self.input_dim, device=device)
        
        # Create padding mask: 1 = valid, 0 = padding
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < input_lengths.unsqueeze(1)).float()  # [batch, seq_len]
        
        # ===================================================================
        # PHASE 2: DECODE - Generate sequence of pointers
        # ===================================================================
        pointers = []
        log_probs_list = []
        attention_weights_list = []
        
        # Mask to track which elements we've already selected
        selection_mask = mask.clone()
        
        for step in range(max_steps):
            # Update decoder state
            decoder_hidden, decoder_cell = self.decoder_cell(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            decoder_hidden = self.dropout(decoder_hidden)
            
            # ===============================================================
            # ATTENTION: Compute pointing scores
            # ===============================================================
            # The heart of Pointer Networks!
            #
            # For each input element, compute how "relevant" it is
            # given the current decoder state.
            #
            # Formula: e_i = v^T * tanh(W1*h_i + W2*s_t)
            # where h_i = encoder output, s_t = decoder state
            
            # Project encoder outputs: [batch, seq_len, hidden_dim]
            encoder_proj = self.W1(encoder_outputs)
            
            # Project decoder state: [batch, hidden_dim] -> [batch, 1, hidden_dim]
            decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
            
            # Combine and apply tanh: [batch, seq_len, hidden_dim]
            combined = torch.tanh(encoder_proj + decoder_proj)
            
            # Project to scalar scores: [batch, seq_len, 1] -> [batch, seq_len]
            scores = self.v(combined).squeeze(-1)
            
            # Apply temperature (for exploration during training)
            scores = scores / temperature
            
            # Mask out padding and already-selected elements
            scores = scores.masked_fill(selection_mask == 0, float('-inf'))
            
            # Convert scores to probabilities
            probs = F.softmax(scores, dim=-1)  # [batch, seq_len]
            log_probs = F.log_softmax(scores, dim=-1)
            
            # ===============================================================
            # SELECT: Choose which element to point to
            # ===============================================================
            if teacher_forcing is not None and step < teacher_forcing.size(1):
                # Training: Use ground truth pointer
                pointer = teacher_forcing[:, step]
            else:
                # Inference: Select greedily (argmax)
                pointer = torch.argmax(probs, dim=-1)  # [batch]
            
            # Store results
            pointers.append(pointer)
            log_probs_list.append(log_probs.gather(1, pointer.unsqueeze(1)))
            attention_weights_list.append(probs)
            
            # Update selection mask (prevent selecting same element twice)
            selection_mask = selection_mask.scatter(1, pointer.unsqueeze(1), 0)
            
            # ===============================================================
            # NEXT INPUT: Use the selected element
            # ===============================================================
            # This is key! The next decoder input is the element we just pointed to.
            # Gather the selected elements from the input
            decoder_input = inputs[torch.arange(batch_size, device=device), pointer]
        
        # Stack results
        pointers = torch.stack(pointers, dim=1)  # [batch, max_steps]
        log_probs = torch.cat(log_probs_list, dim=1)  # [batch, max_steps]
        
        return pointers, log_probs, attention_weights_list


class SetEncoder(nn.Module):
    """
    Order-invariant encoder for sets.
    
    This encoder ensures that the representation doesn't depend on
    the order of input elements. Think of it like a bag of items -
    whether you add milk then eggs, or eggs then milk, the bag
    contains the same stuff!
    
    Key property: encoder([a, b, c]) = encoder([c, a, b])
    
    Achieved by:
    1. Self-attention WITHOUT positional encodings
    2. Permutation-invariant aggregation (mean/max)
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
        
        # Stack of Transformer encoder layers (NO positional encoding!)
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
        """
        Encode a set of elements in an order-invariant way.
        
        Args:
            x: [batch, set_size, input_dim]
            mask: [batch, set_size] - True for padding positions
            
        Returns:
            encoded: [batch, set_size, hidden_dim]
        """
        # Embed inputs
        x = self.embedding(x)
        
        # Apply self-attention layers
        # CRITICAL: No positional encodings added!
        # This ensures order invariance.
        x = self.transformer(x, src_key_padding_mask=mask)
        
        x = self.norm(x)
        
        return x


class ReadProcessWrite(nn.Module):
    """
    Complete Read-Process-Write framework for set-to-sequence problems.
    
    This is the full architecture from the paper!
    
    Three phases:
    1. READ: Encode input set (order-invariant)
    2. PROCESS: Compute set-level representation
    3. WRITE: Generate output sequence with Pointer Network
    
    Example tasks:
    - Sorting: {5,2,9,1} → [1,2,5,9]
    - Convex Hull: {points} → [boundary points in order]
    - TSP: {cities} → [optimal tour]
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
            # Phase 1: READ with order-invariant encoder
            self.encoder = SetEncoder(input_dim, hidden_dim, **kwargs)
            self.pointer_net = PointerNetwork(hidden_dim, hidden_dim, **kwargs)
        else:
            # Baseline: Regular Pointer Network (order-aware encoder)
            self.pointer_net = PointerNetwork(input_dim, hidden_dim, **kwargs)
    
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Full Read-Process-Write forward pass.
        
        Args:
            inputs: [batch, set_size, input_dim]
            input_lengths: [batch]
            max_steps: Number of output steps
            **kwargs: Passed to pointer network (teacher_forcing, etc.)
            
        Returns:
            pointers: [batch, max_steps]
            log_probs: [batch, max_steps]
            attention_weights: List of attention distributions
        """
        if self.use_set_encoder:
            # Phase 1: READ - Encode set in order-invariant way
            encoded = self.encoder(inputs)
            
            # Phase 2: PROCESS - (implicit in encoder)
            
            # Phase 3: WRITE - Generate output with pointer network
            pointers, log_probs, attentions = self.pointer_net(
                encoded, input_lengths, max_steps, **kwargs
            )
        else:
            # Baseline: Direct pointer network
            pointers, log_probs, attentions = self.pointer_net(
                inputs, input_lengths, max_steps, **kwargs
            )
        
        return pointers, log_probs, attentions


def compute_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute negative log-likelihood loss for pointer network.
    
    Args:
        log_probs: [batch, seq_len] - Log probabilities from model
        targets: [batch, seq_len] - Ground truth pointers
        
    Returns:
        loss: Scalar tensor
    """
    # Gather log probabilities of target pointers
    # This is like asking: "How confident were we in the correct pointers?"
    target_log_probs = log_probs.gather(1, targets)
    
    # Negative log-likelihood (we want to maximize log prob = minimize negative log prob)
    loss = -target_log_probs.mean()
    
    return loss


def accuracy(pointers: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        pointers: [batch, seq_len] - Predicted pointers
        targets: [batch, seq_len] - Ground truth pointers
        
    Returns:
        accuracy: Percentage of sequences that are 100% correct
    """
    # Check if entire sequence matches
    correct = (pointers == targets).all(dim=1).float()
    return correct.mean().item()


if __name__ == "__main__":
    # Demo: Sorting with Pointer Network
    print("🎯 Pointer Network Demo: Sorting")
    print("=" * 60)
    
    # Create a simple sorting example
    batch_size = 4
    set_size = 5
    input_dim = 1  # Just the number itself
    hidden_dim = 128
    
    # Generate random numbers to sort
    values = torch.rand(batch_size, set_size, input_dim)
    print(f"Input values (first example):")
    print(values[0].squeeze().tolist())
    
    # Ground truth: sorted indices
    _, sorted_indices = torch.sort(values.squeeze(-1), dim=1)
    print(f"Ground truth sorted indices:")
    print(sorted_indices[0].tolist())
    
    # Create model
    model = ReadProcessWrite(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        use_set_encoder=True,
        num_heads=4,
        num_layers=2
    )
    
    # Forward pass
    lengths = torch.full((batch_size,), set_size)
    pointers, log_probs, attentions = model(
        values, lengths, max_steps=set_size
    )
    
    print(f"\nModel predictions (before training):")
    print(pointers[0].tolist())
    
    print(f"\nAttention weights (first decoding step, first example):")
    print([f"{w:.3f}" for w in attentions[0][0].tolist()])
    
    print(f"\n✅ Model created successfully!")
    print(f"   - Input dim: {input_dim}")
    print(f"   - Hidden dim: {hidden_dim}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
