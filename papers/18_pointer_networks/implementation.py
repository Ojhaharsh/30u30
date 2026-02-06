"""
Pointer Networks (Ptr-Net) Implementation
Standardized for Day 18 of 30u30

A complete, educational implementation of Pointer Networks in PyTorch.
This model solves tasks where the output indices correspond to input positions,
such as sorting, convex hull, and the traveling salesman problem.

Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerAttention(nn.Module):
    """
    Pointer Attention Mechanism (Vinyals et al., 2015).
    
    Instead of blending inputs into a context vector, this module uses the
    attention scores directly as selection probabilities.
    
    Formula (Equation 3):
    u_j^i = v^T * tanh(W_1 * e_j + W_2 * d_i)
    p(C_i | C_1, ..., C_{i-1}, P) = softmax(u^i)
    
    Args:
        hidden_size (int): Dimension of encoder and decoder hidden states.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # W1 transforms encoder states (Keys)
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # W2 transforms decoder state (Query)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        # v projects the combined energy into a single score
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Compute pointer log-probabilities.
        
        Args:
            decoder_hidden (Tensor): Current decoder state (batch, hidden_size)
            encoder_outputs (Tensor): All encoder states (batch, seq_len, hidden_size)
            mask (Tensor, optional): Boolean mask for invalid positions (batch, seq_len)
            
        Returns:
            log_probs (Tensor): Log-selection probabilities (batch, seq_len)
        """
        # === Step 1: Project states ===
        # (batch, 1, hidden_size)
        decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
        # (batch, seq_len, hidden_size)
        encoder_proj = self.W1(encoder_outputs)
        
        # === Step 2: Additive Attention (Eq. 3) ===
        # (batch, seq_len, hidden_size)
        combined = torch.tanh(encoder_proj + decoder_proj)
        
        # === Step 3: Extract scores ===
        # (batch, seq_len)
        scores = self.v(combined).squeeze(-1)
        
        # === Step 4: Apply optional mask ===
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        return F.log_softmax(scores, dim=-1)

class PointerNetwork(nn.Module):
    """
    Pointer Network for sequence-to-sequence pointing.
    
    Architecture:
    - Input projection: Raw features to hidden space
    - Encoder (LSTM): Processes input sequence
    - Decoder (LSTM): Generates output pointers autoregressively
    - Pointer Attention: Selection head
    
    Args:
        input_size (int): Number of features per input item (e.g., 1 for sorting, 2 for coords).
        hidden_size (int): Dimension of the embeddings and LSTM hidden states.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Basic projection for input coordinates/features
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.pointer = PointerAttention(hidden_size)
        
    def forward(self, inputs, trg_len=None):
        """
        Forward pass with greedily selected pointers.
        
        Args:
            inputs (Tensor): Input sequence features (batch, seq_len, input_size)
            trg_len (int, optional): Output steps to generate. Defaults to input seq_len.
            
        Returns:
            log_pointers (Tensor): Sequence of selection probs (batch, trg_len, seq_len)
        """
        batch_size, seq_len, _ = inputs.size()
        if trg_len is None:
            trg_len = seq_len
            
        # === Step 1: Project and Encode ===
        x = self.input_proj(inputs)
        encoder_outputs, (h, c) = self.encoder(x)
        
        # === Step 2: Initialize Decoder ===
        # Initial decoder input (zeros) and hidden state (from encoder)
        decoder_input = torch.zeros(batch_size, 1, self.hidden_size, device=inputs.device)
        decoder_hidden = (h, c)
        
        all_log_pointers = []
        # Mask can be used to prevent re-selection in tasks like TSP
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=inputs.device)
        
        # === Step 3: Autoregressive Decoding ===
        for _ in range(trg_len):
            # 3a. Step decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # 3b. Compute pointer probabilities
            log_probs = self.pointer(decoder_output.squeeze(1), encoder_outputs, mask)
            all_log_pointers.append(log_probs)
            
            # 3c. Select next input (Greedy selection for demo)
            top_idx = log_probs.argmax(dim=-1)
            
            # 3d. Update decoder input with the selected item's representation
            decoder_input = torch.stack([encoder_outputs[i, idx] for i, idx in enumerate(top_idx)]).unsqueeze(1)
            
        return torch.stack(all_log_pointers, dim=1)

if __name__ == "__main__":
    print("Pointer Network Implementation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerNetwork(input_size=1, hidden_size=64).to(device)
    
    # 2 random sequences of 5 numbers for verification
    dummy_input = torch.rand(2, 5, 1).to(device)
    output = model(dummy_input)
    
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape} (batch, seq_len, features)")
    print(f"Output shape: {output.shape} (batch, steps, pointers)")
    print("\n[OK] Implementation verification complete.")
