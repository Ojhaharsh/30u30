import torch
import torch.nn as nn

class LSTMUpdate(nn.Module):
    def __init__(self, input_size, state_size):
        super().__init__()
        # We will assume input to the gates is the concatenation of [input, state]
        feature_dim = input_size + state_size
        
        # TODO: Define the linear layers for Forget, Input, and Candidate gates
        # Each should output a tensor of size 'state_size'
        self.gate_f = # YOUR CODE HERE
        self.gate_i = # YOUR CODE HERE
        self.gate_c = # YOUR CODE HERE
        
    def forward(self, input_vec, state):
        """
        Args:
            input_vec: (batch, slots, input_size) - this is the "new stuff" (attended info)
            state: (batch, slots, state_size) - this is the "old stuff" (previous memory)
        """
        # Concatenate along the last dimension
        combined = torch.cat([input_vec, state], dim=-1)
        
        # TODO: Calculate gate activations
        # Remember: Sigmoid for f and i, Tanh for candidate
        f = # YOUR CODE HERE
        i = # YOUR CODE HERE
        c = # YOUR CODE HERE
        
        # TODO: Compute next state
        # next_state = f * state + i * candidate
        next_state = # YOUR CODE HERE
        
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
