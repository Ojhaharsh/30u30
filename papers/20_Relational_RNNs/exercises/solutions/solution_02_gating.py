import torch
import torch.nn as nn

class LSTMUpdate(nn.Module):
    def __init__(self, input_size, state_size):
        super().__init__()
        # We will assume input to the gates is the concatenation of [input, state]
        feature_dim = input_size + state_size
        
        # Define the linear layers for Forget, Input, and Candidate gates
        self.gate_f = nn.Linear(feature_dim, state_size)
        self.gate_i = nn.Linear(feature_dim, state_size)
        self.gate_c = nn.Linear(feature_dim, state_size)
        
    def forward(self, input_vec, state):
        """
        Args:
            input_vec: (batch, slots, input_size) - this is the "new stuff" (attended info)
            state: (batch, slots, state_size) - this is the "old stuff" (previous memory)
        """
        # Concatenate along the last dimension
        combined = torch.cat([input_vec, state], dim=-1)
        
        # Calculate gate activations
        f = torch.sigmoid(self.gate_f(combined))
        i = torch.sigmoid(self.gate_i(combined))
        c = torch.tanh(self.gate_c(combined))
        
        # Compute next state
        next_state = f * state + i * c
        
        return next_state

def test_gating():
    print("Testing LSTM Gating...")
    batch, slots, dim = 2, 4, 16
    updater = LSTMUpdate(dim, dim) # Input size = state size for simplicity
    
    x = torch.randn(batch, slots, dim)
    m = torch.randn(batch, slots, dim)
    
    m_new = updater(x, m)
    
    assert m_new.shape == (batch, slots, dim)
    print("Test Passed!")

if __name__ == "__main__":
    test_gating()
