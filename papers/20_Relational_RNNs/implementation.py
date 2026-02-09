"""
Day 20: Relational Recurrent Neural Networks (RMC) - Implementation
=====================================================================

A complete PyTorch implementation of the Relational Memory Core (RMC), covering:
- Multi-Head Dot Product Attention (MHDPA) over memory slots
- LSTM-style Gating for memory updates
- Comparison: RMC vs. Standard LSTM Baseline

Paper: "Relational recurrent neural networks"
        by Adam Santoro et al. (2018)

Think of this as: An RNN where the hidden state is a "team" of vectors 
that talk to each other before making a decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelationalMemory(nn.Module):
    """
    Relational Memory Core (RMC) from Santoro et al. (2018).
    
    A recurrent module that uses self-attention to allow memory slots to interact
    at each time step.
    """
    def __init__(self, mem_slots, mem_size, input_size, num_heads=4, num_blocks=1, dropout=0.1):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # ---------------------------------------------------------------------
        # 1. Input Projection
        # ---------------------------------------------------------------------
        # Implementation Note:
        # Per Santoro et al. (2018), Section 2: "we simply concatenate this 
        # projected input to each row of the memory matrix."
        # This implies: M_in = [M_{t-1}; x_projected_tiled]
        #
        # We project input to the same size as memory slots to allow additive 
        # or concatenative interaction.
        self.input_projector = nn.Linear(input_size, mem_size)
        
        # ---------------------------------------------------------------------
        # 2. Attention Mechanism (Self-Attention over Memory Slots)
        # ---------------------------------------------------------------------
        # The input to attention is the concatenation [Memory; Input_Tiled].
        # Feature dimension = mem_size (from memory) + mem_size (from input)
        feature_dim_in = 2 * mem_size 
        
        self.attention_key_size = mem_size // num_heads 
        self.attention_value_size = mem_size // num_heads

        # Q, K, V Projections
        self.q_proj = nn.Linear(feature_dim_in, mem_size)
        self.k_proj = nn.Linear(feature_dim_in, mem_size)
        self.v_proj = nn.Linear(feature_dim_in, mem_size)
        
        self.output_proj = nn.Linear(mem_size, mem_size) 
        
        self.layer_norm_mem = nn.LayerNorm(mem_size)
        self.layer_norm_output = nn.LayerNorm(mem_size)
        
        # ---------------------------------------------------------------------
        # 3. Gating Mechanism (LSTM-style Input/Forget Gates)
        # ---------------------------------------------------------------------
        # Implementation Note:
        # Standard RMC implementations (e.g., DeepMind Sonnet) apply gating 
        # to the ATTENDED memory and the OLD memory.
        #
        # Update rule:
        # m_t = f_t * m_{prev} + i_t * tanh(m_attended)
        
        self.gate_forget = nn.Linear(feature_dim_in, mem_size)
        self.gate_input = nn.Linear(feature_dim_in, mem_size)
        
        # Initialize forget gate bias to 1.0 to encourage remembering long-term dependencies
        nn.init.constant_(self.gate_forget.bias, 1.0)
        # No output gate in basic RMC, just next memory state.
        
        # State initialization
        # "We treat the initial memory state as a learnable parameter."
        self.initial_memory = nn.Parameter(torch.randn(1, mem_slots, mem_size) * 0.01)

    def forward(self, input_step, memory_state=None):
        """
        One step of RMC.
        
        input_step: (batch, input_size)
        memory_state: (batch, mem_slots, mem_size)
        """
        batch_size = input_step.size(0)
        
        if memory_state is None:
            memory_state = self.initial_memory.expand(batch_size, -1, -1)
            
        # ---------------------------------------------------------------------
        # 1. Project Input and Tile
        # ---------------------------------------------------------------------
        # input_proj: (batch, mem_size)
        input_proj = self.input_projector(input_step)
        
        # Tile: (batch, mem_slots, mem_size)
        input_tiled = input_proj.unsqueeze(1).expand(-1, self.mem_slots, -1)
        
        # ---------------------------------------------------------------------
        # 2. Augmented Memory (Memory + Input)
        # ---------------------------------------------------------------------
        # memory_aug: (batch, mem_slots, 2*mem_size)
        memory_aug = torch.cat([memory_state, input_tiled], dim=-1)
        
        # ---------------------------------------------------------------------
        # 3. Multi-Head Self-Attention
        # ---------------------------------------------------------------------
        # Create Q, K, V from [M_old; x]
        q = self.q_proj(memory_aug)
        k = self.k_proj(memory_aug)
        v = self.v_proj(memory_aug)
        
        # Reshape for multi-head: (batch, heads, slots, head_dim)
        head_dim = self.mem_size // self.num_heads
        
        q = q.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        probs = F.softmax(scores, dim=-1)
        
        # Weighted sum & Recombine
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, self.mem_slots, self.mem_size)
        
        # Linear + Residual + LayerNorm
        output = self.output_proj(output)
        m_attended = output + memory_state
        m_attended = self.layer_norm_mem(m_attended)
        
        # ---------------------------------------------------------------------
        # 4. Gated Update
        # ---------------------------------------------------------------------
        # Compute gates using the original augmented memory state [M_old; x]
        f_gate = torch.sigmoid(self.gate_forget(memory_aug))
        i_gate = torch.sigmoid(self.gate_input(memory_aug))
        
        # Candidate memory state (tanh nonlinearity)
        # We use the attended memory 'output' as the basis for the new content
        candidate = torch.tanh(output) 
        
        # Final LSTM-style update:
        # m_new = f * m_old + i * candidate
        next_memory = f_gate * memory_state + i_gate * candidate
        
        # Final LSTM-style update:
        # m_new = f * m_old + i * candidate
        next_memory = f_gate * memory_state + i_gate * candidate
        
        return next_memory, probs

class RelationalRNN(nn.Module):
    """
    Wrapper for sequence processing using RMC.
    """
    def __init__(self, input_size, mem_slots, mem_size, output_size, num_heads=4):
        super().__init__()
        self.rmc = RelationalMemory(mem_slots, mem_size, input_size, num_heads)
        self.out = nn.Linear(mem_slots * mem_size, output_size)
    
    def forward(self, x, return_attention=False):
        # x: (batch, seq, input)
        batch_size = x.size(0)
        memory = None
        attention_history = []
        
        for t in range(x.size(1)):
            input_step = x[:, t, :]
            memory, probs = self.rmc(input_step, memory)
            if return_attention:
                attention_history.append(probs)
        
        # Flatten memory for output
        # (batch, slots*size)
        flat_memory = memory.view(batch_size, -1)
        output = self.out(flat_memory)
        
        if return_attention:
            # Stack history: (batch, seq, heads, slots, slots)
            return output, torch.stack(attention_history, dim=1)
            
        return output

class StandardLSTM(nn.Module):
    """
    Baseline LSTM model for comparison.
    Uses similar parameter count to RMC for fair(ish) comparison.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq, input)
        # LSTM returns (output, (h_n, c_n))
        # output: (batch, seq, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        
        # We only care about the final state for the N-th Farthest task
        # h_n: (num_layers, batch, hidden_size)
        final_state = h_n[-1]
        
        return self.out(final_state)
