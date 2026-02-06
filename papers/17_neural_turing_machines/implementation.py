"""
Neural Turing Machine (NTM) Implementation
==========================================

This is a complete implementation of the Neural Turing Machine architecture 
described in Graves et al. (2014).

Key Principles:
1. Differentiability: Every operation (Read, Write, Address) is differentiable, 
   allowing end-to-end training via backpropagation.
2. Decoupled Memory: Unlike standard RNNs, the memory capacity (N x M) is 
   independent of the controller's parameter count.
3. Addressing: Uses both Content-based (similarity) and Location-based (shift) 
   addressing mechanisms.

Traceability:
- Section 3: Memory and Read/Write Heads
- Section 3.3.1: The Addressing Pipeline
- Section 4: Experimental results and training stability

Based on the original paper "Neural Turing Machines" (arxiv.org/abs/1410.5401)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTMMemory(nn.Module):
    """
    The external memory matrix (Section 3).
    Stored as a matrix of size N x M (locations x vector dimension).
    """
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        
        # Section 3.2: Memory initialization.
        # We use a learned buffer for the starting state of memory.
        # Graves mentions that initial memory can be learned.
        self.register_buffer('init_memory', torch.full((N, M), 1e-6))

    def reset(self, batch_size):
        """Initializes memory for a new sequence."""
        return self.init_memory.clone().repeat(batch_size, 1, 1)


class NTMHead(nn.Module):
    """
    The Read/Write Head interface (Section 3.1 & 3.2).
    Implements the 4-step Addressing Pipeline (Section 3.3.1) to generate w_t.
    """
    def __init__(self, memory, controller_size, is_read=True):
        super().__init__()
        self.N = memory.N
        self.M = memory.M
        self.is_read = is_read
        
        # Addressing parameters (Section 3.3.1):
        # k (M): key vector (Eq 5)
        # beta (1): key strength (Eq 5)
        # g (1): interpolation gate (Eq 7)
        # s (3): shift weighting for circular convolution (Eq 8)
        # gamma (1): sharpening factor (Eq 9)
        self.addr_size = self.M + 1 + 1 + 3 + 1
        
        # Write heads require additional vectors (Section 3.3):
        # e (M): erase vector (Eq 3)
        # a (M): add vector (Eq 4)
        if not is_read:
            self.addr_size += 2 * self.M
            
        self.fc = nn.Linear(controller_size, self.addr_size)
        
        # Learned initial weighting (Section 2)
        # Usually initializes focus at the first memory location.
        self.register_buffer('init_w', torch.zeros(self.N))
        self.init_w[0] = 1.0 

    def reset(self, batch_size):
        return self.init_w.clone().repeat(batch_size, 1)

    def _addressing(self, controller_out, prev_w, memory):
        """
        Calculates the addressing weighting w_t.
        Maps Equations 5 through 9 of the paper.
        """
        # 0. Generate all addressing parameters from the controller hidden state
        out = self.fc(controller_out)
        
        ptr = 0
        k = out[:, ptr:ptr+self.M]; ptr += self.M
        # beta/gamma must be positive (Section 3.3.1). Softplus ensures this.
        beta = F.softplus(out[:, ptr:ptr+1]); ptr += 1
        g = torch.sigmoid(out[:, ptr:ptr+1]); ptr += 1
        # Shift s is a distribution over three values [-1, 0, 1] (Eq 8)
        s = F.softmax(out[:, ptr:ptr+3], dim=1); ptr += 3
        gamma = 1 + F.softplus(out[:, ptr:ptr+1]); ptr += 1
        
        erase = None
        add = None
        if not self.is_read:
            erase = torch.sigmoid(out[:, ptr:ptr+self.M]); ptr += self.M
            add = out[:, ptr:ptr+self.M]

        # 1. Content Addressing (Eq 5 & 6)
        # Focuses based on what the information IS (associative retrieval).
        norm_k = k.norm(p=2, dim=1, keepdim=True) + 1e-8
        norm_mem = memory.norm(p=2, dim=2) + 1e-8
        sim = torch.matmul(memory, k.unsqueeze(2)).squeeze(2) / (norm_k * norm_mem)
        w_c = F.softmax(beta * sim, dim=1)

        # 2. Interpolation (Eq 7)
        # Blends new search (w_c) with previous focus (prev_w).
        # This allows the NTM to "stay" at a location or move to a new one.
        w_g = g * w_c + (1 - g) * prev_w

        # 3. Convolutional Shift (Eq 8)
        # Performs circular convolution to allow relative movement.
        # Necessary for iterating through sequences (the Copy Task).
        w_s = torch.zeros_like(w_g)
        for i in range(self.N):
            # Circular shift: s[-1]*w[i+1] + s[0]*w[i] + s[1]*w[i-1]
            w_s[:, i] = (
                s[:, 0] * torch.roll(w_g, shifts=1, dims=1)[:, i] +
                s[:, 1] * w_g[:, i] +
                s[:, 2] * torch.roll(w_g, shifts=-1, dims=1)[:, i]
            )

        # 4. Sharpening (Eq 9)
        # Without sharpening, the focus becomes more blurry at every shift step.
        # Sharpening "peaks" the distribution to maintain focus.
        w_pow = w_s ** gamma
        w = w_pow / (w_pow.sum(dim=1, keepdim=True) + 1e-8)

        return w, erase, add

    def forward(self, controller_out, prev_w, memory_state):
        w, erase, add = self._addressing(controller_out, prev_w, memory_state)
        
        if self.is_read:
            # Read Operation (Eq 2): Weighted sum of memory rows
            read_vec = torch.matmul(w.unsqueeze(1), memory_state).squeeze(1)
            return read_vec, w
        else:
            return w, erase, add


class NTM(nn.Module):
    """
    Neural Turing Machine Wrapper (Section 2).
    Connects a Controller (LSTM) to Memory via Read and Write Heads.
    """
    def __init__(self, input_size, output_size, controller_size, N, M, num_reads=1, num_writes=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.N = N
        self.M = M
        
        self.memory = NTMMemory(N, M)
        
        # Section 2: Controller receives both external input AND previous read vectors.
        # This creates the "thinking" loop that includes memory state.
        self.controller = nn.LSTMCell(input_size + num_reads * M, controller_size)
        
        self.read_heads = nn.ModuleList([NTMHead(self.memory, controller_size, is_read=True) for _ in range(num_reads)])
        self.write_heads = nn.ModuleList([NTMHead(self.memory, controller_size, is_read=False) for _ in range(num_writes)])
        
        # Final output layer projections
        self.fc_out = nn.Linear(controller_size + num_reads * M, output_size)
        
        # Learned initial hidden states for the controller
        self.register_buffer('h_bias', torch.zeros(controller_size))
        self.register_buffer('c_bias', torch.zeros(controller_size))

    def reset(self, batch_size):
        """Prepares the NTM for a new sequence (Section 2)."""
        memory_state = self.memory.reset(batch_size)
        h = self.h_bias.clone().repeat(batch_size, 1)
        c = self.c_bias.clone().repeat(batch_size, 1)
        
        # Reset heads to their initial weighting (usually location 0)
        read_ws = [head.reset(batch_size) for head in self.read_heads]
        write_ws = [head.reset(batch_size) for head in self.write_heads]
        
        # Initial read vectors are usually zero
        prev_reads = [torch.zeros(batch_size, self.M, device=memory_state.device) for _ in range(len(self.read_heads))]
        
        return memory_state, (h, c), read_ws, write_ws, prev_reads

    def forward(self, x, state):
        memory_state, (h, c), read_ws, write_ws, prev_reads = state
        
        # 1. Controller Pass (Section 2)
        # Concatenate input with what we read in the PREVIOUS step.
        inp = torch.cat([x] + prev_reads, dim=1)
        h, c = self.controller(inp, (h, c))
        
        # 2. Reading Step (Eq 2)
        new_read_ws = []
        new_reads = []
        for i, head in enumerate(self.read_heads):
            r, w = head(h, read_ws[i], memory_state)
            new_reads.append(r)
            new_read_ws.append(w)
            
        # 3. Writing Step (Eq 3 & 4)
        new_write_ws = []
        for i, head in enumerate(self.write_heads):
            w, erase, add = head(h, write_ws[i], memory_state)
            new_write_ws.append(w)
            
            # Erase Step (Eq 3): Remove old info
            erase_term = torch.matmul(w.unsqueeze(2), erase.unsqueeze(1))
            memory_state = memory_state * (1 - erase_term)
            
            # Add Step (Eq 4): Inject new info
            add_term = torch.matmul(w.unsqueeze(2), add.unsqueeze(1))
            memory_state = memory_state + add_term
            
        # 4. Emit output (Section 2)
        # Output is based on controller state and currently read information.
        out_inp = torch.cat([h] + new_reads, dim=1)
        out = self.fc_out(out_inp)
        
        new_state = (memory_state, (h, c), new_read_ws, new_write_ws, new_reads)
        return out, new_state

if __name__ == "__main__":
    # Smoke test: Initialize a small NTM and run a single step
    ntm = NTM(input_size=8, output_size=8, controller_size=100, N=128, M=20)
    print(f"NTM initialized with {sum(p.numel() for p in ntm.parameters())} parameters.")
    
    batch_size = 4
    state = ntm.reset(batch_size)
    x = torch.randn(batch_size, 8)
    
    output, _ = ntm(x, state)
    print(f"Forward pass completed. Output shape: {output.shape}")
