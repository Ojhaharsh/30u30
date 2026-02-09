import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# We can import from previous exercises, but let's keep it self-contained for simplicity
class RelationalMemoryStep(nn.Module):
    def __init__(self, mem_slots, mem_size, input_size, num_heads=4):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.num_heads = num_heads
        
        # 1. Input Projector: Input -> Memory Size
        self.input_proj = nn.Linear(input_size, mem_size)
        
        # 2. Attention Projections
        # Input to attention is [Memory; Input_Tiled] (dim = 2 * mem_size)
        feature_dim = 2 * mem_size
        
        self.q_proj = nn.Linear(feature_dim, mem_size)
        self.k_proj = nn.Linear(feature_dim, mem_size)
        self.v_proj = nn.Linear(feature_dim, mem_size)
        
        self.output_proj = nn.Linear(mem_size, mem_size)
        self.layer_norm = nn.LayerNorm(mem_size)
        
        # 3. Gating (Input, Forget)
        # We'll use a simple LSTM update
        self.gate_f = nn.Linear(feature_dim, mem_size)
        self.gate_i = nn.Linear(feature_dim, mem_size)
        
    def forward(self, input_step, memory_state):
        batch_size = input_step.size(0)
        
        # TODO: Project input and tile it to match memory slots
        # input_proj: (batch, mem_size)
        # input_tiled: (batch, slots, mem_size)
        input_proj = # YOUR CODE HERE
        input_tiled = input_proj.unsqueeze(1).expand(-1, self.mem_slots, -1)
        
        # TODO: Concatenate memory and tiled input
        # memory_aug: (batch, slots, 2*mem_size)
        memory_aug = # YOUR CODE HERE
        
        # TODO: Perform Multi-Head Attention on memory_aug
        # For this exercise, you can use PyTorch's F.multi_head_attention_forward OR
        # just implement the projection -> score -> softmax -> value logic.
        # Let's simplify and just do the projections and matmul manually to reinforce Ex 1.
        
        q = self.q_proj(memory_aug)
        k = self.k_proj(memory_aug)
        v = self.v_proj(memory_aug)
        
        # Reshape for multi-head (batch, heads, slots, dim/heads)
        head_dim = self.mem_size // self.num_heads
        q = q.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        
        # Attention Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.mem_slots, self.mem_size)
        
        # Output projection + Residual (from memory) + Layer Norm
        output = self.output_proj(attended)
        m_attended = self.layer_norm(output + memory_state)
        
        # TODO: Gating Update
        # Calculate f_gate and i_gate using memory_aug
        # Calculate candidate using tanh(output) or tanh(m_attended)
        # Update memory
        
        f_gate = # YOUR CODE HERE
        i_gate = # YOUR CODE HERE
        
        next_memory = # YOUR CODE HERE
        
        return next_memory

def test_rmc_step():
    print("Testing One RMC Step...")
    batch, slots, mem_dim, input_dim = 2, 4, 32, 10
    rmc = RelationalMemoryStep(slots, mem_dim, input_dim)
    
    x = torch.randn(batch, input_dim)
    m = torch.randn(batch, slots, mem_dim)
    
    m_new = rmc(x, m)
    
    assert m_new.shape == (batch, slots, mem_dim)
    print("Test Passed!")

if __name__ == "__main__":
    test_rmc_step()
