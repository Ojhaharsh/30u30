"""
Exercise 1: Build LSTM from Scratch
====================================

Goal: Implement a complete LSTM cell with all 4 gates and backpropagation.

Your Task:
- Fill in the TODOs below to complete the LSTM implementation
- Test your implementation on a simple sequence
- Compare with the reference implementation

Learning Objectives:
1. Understand how each gate works mathematically
2. See how cell state and hidden state interact
3. Learn how gradients flow backward through the LSTM
4. Appreciate why LSTMs solve vanishing gradients

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import numpy as np


class LSTMFromScratch:
    """
    A Long Short-Term Memory (LSTM) network implemented from scratch.
    
    Architecture:
        - Forget gate: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
        - Input gate: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
        - Cell candidate: C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)
        - Output gate: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
        - Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        - Hidden state: h_t = o_t ⊙ tanh(C_t)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize LSTM with random weights.
        
        Args:
            input_size: Size of input vectors
            hidden_size: Number of LSTM units
            output_size: Size of output vectors
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # TODO 1: Initialize weight matrices for forget gate
        # Hint: Shape should be (hidden_size, hidden_size + input_size)
        # Use Xavier initialization: np.random.randn() * 0.01
        self.Wf = None  # TODO: Initialize
        self.bf = None  # TODO: Initialize to np.ones(hidden_size) - why ones?
        
        # TODO 2: Initialize weight matrices for input gate
        self.Wi = None  # TODO: Initialize
        self.bi = None  # TODO: Initialize to np.zeros(hidden_size)
        
        # TODO 3: Initialize weight matrices for cell candidate
        self.Wc = None  # TODO: Initialize
        self.bc = None  # TODO: Initialize to np.zeros(hidden_size)
        
        # TODO 4: Initialize weight matrices for output gate
        self.Wo = None  # TODO: Initialize
        self.bo = None  # TODO: Initialize to np.zeros(hidden_size)
        
        # TODO 5: Initialize output layer weights
        self.Wy = None  # TODO: Shape (output_size, hidden_size)
        self.by = None  # TODO: Shape (output_size,)
        
    def sigmoid(self, x):
        """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        Forward pass through LSTM for a sequence.
        
        Args:
            inputs: List of input indices (length T)
            targets: List of target indices (length T)
            h_prev: Previous hidden state (hidden_size,)
            C_prev: Previous cell state (hidden_size,)
            
        Returns:
            loss: Cross-entropy loss over the sequence
        """
        # Storage for intermediate values (needed for backward pass)
        self.inputs_cache = []
        self.h_states = [h_prev]
        self.C_states = [C_prev]
        self.f_gates = []
        self.i_gates = []
        self.C_tilde_cache = []
        self.o_gates = []
        self.y_probs = []
        
        loss = 0.0
        
        for t, (input_idx, target_idx) in enumerate(zip(inputs, targets)):
            # Create one-hot encoded input
            x = np.zeros(self.input_size)
            x[input_idx] = 1.0
            self.inputs_cache.append(x)
            
            # TODO 6: Concatenate previous hidden state and current input
            # Shape: (hidden_size + input_size,)
            concat = None  # TODO: Implement
            
            # TODO 7: Compute forget gate
            # f_t = sigmoid(W_f · concat + b_f)
            f_t = None  # TODO: Implement
            self.f_gates.append(f_t)
            
            # TODO 8: Compute input gate
            # i_t = sigmoid(W_i · concat + b_i)
            i_t = None  # TODO: Implement
            self.i_gates.append(i_t)
            
            # TODO 9: Compute cell candidate
            # C̃_t = tanh(W_C · concat + b_C)
            C_tilde = None  # TODO: Implement
            self.C_tilde_cache.append(C_tilde)
            
            # TODO 10: Update cell state
            # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            # This is the KEY equation that solves vanishing gradients!
            C_t = None  # TODO: Implement
            self.C_states.append(C_t)
            
            # TODO 11: Compute output gate
            # o_t = sigmoid(W_o · concat + b_o)
            o_t = None  # TODO: Implement
            self.o_gates.append(o_t)
            
            # TODO 12: Compute hidden state
            # h_t = o_t ⊙ tanh(C_t)
            h_t = None  # TODO: Implement
            self.h_states.append(h_t)
            
            # TODO 13: Compute output
            # y = W_y · h_t + b_y
            y = None  # TODO: Implement
            
            # TODO 14: Apply softmax to get probabilities
            # Hint: Use exp and normalize
            probs = None  # TODO: Implement softmax
            self.y_probs.append(probs)
            
            # TODO 15: Compute cross-entropy loss
            # loss = -log(probability of correct class)
            loss += None  # TODO: Implement
            
        return loss
    
    def backward(self):
        """
        Backward pass (Backpropagation Through Time).
        
        Computes gradients for all parameters.
        
        Returns:
            dh_next: Gradient w.r.t. next hidden state
            dC_next: Gradient w.r.t. next cell state
        """
        # Initialize gradients
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)
        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros(self.hidden_size)
        dC_next = np.zeros(self.hidden_size)
        
        # Backward through time
        for t in reversed(range(len(self.inputs_cache))):
            # TODO 16: Gradient from output layer
            dy = self.y_probs[t].copy()
            dy[targets[t]] -= 1  # Softmax + cross-entropy gradient
            
            # TODO 17: Gradient w.r.t. output weights
            dWy += None  # TODO: Implement (outer product)
            dby += None  # TODO: Implement
            
            # TODO 18: Gradient w.r.t. hidden state
            dh = None  # TODO: From output layer + from next time step
            
            # TODO 19: Gradient w.r.t. output gate
            # dh = do ⊙ tanh(C_t) + (gradient from next step)
            # Backprop through: h_t = o_t ⊙ tanh(C_t)
            do = None  # TODO: Implement
            dbo += do
            
            # TODO 20: Gradient w.r.t. cell state
            # From: h_t = o_t ⊙ tanh(C_t)
            dC = None  # TODO: Implement (include dC_next from next step)
            
            # TODO 21: Gradient w.r.t. cell candidate
            # From: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            dC_tilde = None  # TODO: Implement
            dbc += dC_tilde
            
            # TODO 22: Gradient w.r.t. input gate
            # From: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            di = None  # TODO: Implement
            dbi += di
            
            # TODO 23: Gradient w.r.t. forget gate
            # From: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            df = None  # TODO: Implement
            dbf += df
            
            # TODO 24: Gradient w.r.t. previous cell state
            # This is the KEY! Notice it's additive (not multiplicative)
            dC_next = None  # TODO: Implement (gradient flows through forget gate)
            
            # TODO 25: Backprop to weight matrices
            concat = np.concatenate([self.h_states[t], self.inputs_cache[t]])
            dWo += None  # TODO: Implement (outer product)
            dWc += None  # TODO: Implement
            dWi += None  # TODO: Implement
            dWf += None  # TODO: Implement
            
            # TODO 26: Gradient w.r.t. previous hidden state
            dh_next = None  # TODO: Implement (from all 4 gates)
        
        # TODO 27: Apply gradient clipping to prevent explosion
        # Hint: np.clip(grad, -5, 5)
        for dparam in [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # Store gradients
        self.dWf, self.dbf = dWf, dbf
        self.dWi, self.dbi = dWi, dbi
        self.dWc, self.dbc = dWc, dbc
        self.dWo, self.dbo = dWo, dbo
        self.dWy, self.dby = dWy, dby
        
        return dh_next, dC_next
    
    def update_weights(self, learning_rate):
        """Update weights using gradients."""
        # TODO 28: Implement weight updates
        # w = w - learning_rate * dw
        self.Wf -= None  # TODO
        self.bf -= None  # TODO
        # ... continue for all parameters


# ============================================================================
# Testing Your Implementation
# ============================================================================

def test_lstm():
    """Test your LSTM on a simple sequence."""
    print("Testing LSTM implementation...")
    print("=" * 60)
    
    # Simple data: learn to repeat "hello"
    text = "hellohellohello"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    vocab_size = len(chars)
    hidden_size = 10
    
    # Create LSTM
    lstm = LSTMFromScratch(vocab_size, hidden_size, vocab_size)
    
    # Training
    seq_length = 5
    learning_rate = 0.01
    
    h = np.zeros(hidden_size)
    C = np.zeros(hidden_size)
    
    for iteration in range(100):
        # Get random sequence
        start = np.random.randint(0, len(text) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in text[start:start+seq_length]]
        targets = [char_to_idx[ch] for ch in text[start+1:start+seq_length+1]]
        
        # Forward
        loss = lstm.forward(inputs, targets, h, C)
        
        # Backward
        dh, dC = lstm.backward()
        
        # Update
        lstm.update_weights(learning_rate)
        
        # Update states
        h = lstm.h_states[-1].copy()
        C = lstm.C_states[-1].copy()
        
        if iteration % 20 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.4f}")
    
    print("\n✅ If loss is decreasing, your implementation is likely correct!")
    print("Compare with reference implementation in ../implementation.py")


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs (search for 'TODO')")
    print("2. Run this file to test your implementation")
    print("3. Loss should decrease over iterations")
    print("4. Check ../solutions/solution_01_lstm_implementation.py if stuck")
    print("=" * 60 + "\n")
    
    # Uncomment when ready to test
    # test_lstm()
