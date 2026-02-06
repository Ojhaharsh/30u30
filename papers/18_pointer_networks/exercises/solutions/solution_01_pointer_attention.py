"""
Solution 1: Implement the Pointer Attention mechanism
Standardized for Day 18 of 30u30

Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # 1. Project states
        decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
        encoder_proj = self.W1(encoder_outputs)
        
        # 2. Additive Attention (Eq. 3)
        combined = torch.tanh(encoder_proj + decoder_proj)
        
        # 3. Project to scores
        scores = self.v(combined).squeeze(-1)
        
        # 4. Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        return F.log_softmax(scores, dim=-1)
