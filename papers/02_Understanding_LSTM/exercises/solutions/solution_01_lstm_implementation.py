"""
Solution to Exercise 1: Build LSTM from Scratch
================================================

This is the complete solution with all TODOs filled in.

Compare your implementation with this one to check your understanding.
"""

import numpy as np


class LSTMFromScratch:
    """Complete LSTM implementation with detailed comments."""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # SOLUTION 1: Forget gate weights
        # Forget gate decides what to remove from cell state
        # Initialized with small random values (Xavier initialization)
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        # Bias initialized to 1 so LSTM "remembers by default" (important!)
        self.bf = np.ones(hidden_size)
        
        # SOLUTION 2: Input gate weights
        # Input gate decides what new information to add
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bi = np.zeros(hidden_size)
        
        # SOLUTION 3: Cell candidate weights
        # Creates new candidate values to add to cell state
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bc = np.zeros(hidden_size)
        
        # SOLUTION 4: Output gate weights
        # Output gate decides what to reveal from cell state
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bo = np.zeros(hidden_size)
        
        # SOLUTION 5: Output layer weights
        # Maps hidden state to output vocabulary
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros(output_size)
        
    def sigmoid(self, x):
        """Sigmoid activation: maps to [0, 1] range (for gates)."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        Forward pass through LSTM.
        
        This implements the core LSTM equations step by step.
        """
        # Storage for backward pass
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
            # One-hot encode input
            x = np.zeros(self.input_size)
            x[input_idx] = 1.0
            self.inputs_cache.append(x)
            
            # SOLUTION 6: Concatenate h_{t-1} and x_t
            # This is the input to all gates
            concat = np.concatenate([self.h_states[t], x])
            
            # SOLUTION 7: Forget gate
            # Decides what to throw away from cell state
            # Output is between 0 (forget everything) and 1 (keep everything)
            f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            self.f_gates.append(f_t)
            
            # SOLUTION 8: Input gate
            # Decides which new values to add to cell state
            # Output is between 0 (ignore) and 1 (accept)
            i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            self.i_gates.append(i_t)
            
            # SOLUTION 9: Cell candidate
            # Creates new candidate values
            # Uses tanh (not sigmoid) so values are between -1 and 1
            C_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
            self.C_tilde_cache.append(C_tilde)
            
            # SOLUTION 10: Update cell state
            # This is THE KEY EQUATION that solves vanishing gradients!
            # Notice: it's ADDITIVE, not multiplicative
            # C_t = (what to keep from old) + (what to add from new)
            C_t = f_t * self.C_states[t] + i_t * C_tilde
            self.C_states.append(C_t)
            
            # SOLUTION 11: Output gate
            # Decides what parts of cell state to output
            o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            self.o_gates.append(o_t)
            
            # SOLUTION 12: Hidden state
            # The actual output of the LSTM cell
            # h_t = (what to reveal) * (squashed cell state)
            h_t = o_t * np.tanh(C_t)
            self.h_states.append(h_t)
            
            # SOLUTION 13: Compute output logits
            y = np.dot(self.Wy, h_t) + self.by
            
            # SOLUTION 14: Softmax to get probabilities
            # Subtract max for numerical stability
            exp_y = np.exp(y - np.max(y))
            probs = exp_y / np.sum(exp_y)
            self.y_probs.append(probs)
            
            # SOLUTION 15: Cross-entropy loss
            # -log(probability of correct class)
            loss += -np.log(probs[target_idx] + 1e-8)  # Add epsilon to avoid log(0)
            
        return loss
    
    def backward(self):
        """
        Backward pass (BPTT).
        
        This is where gradients flow backward through time.
        Notice how cell state gradient flows almost unchanged!
        """
        # Initialize all gradients to zero
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
        
        # Process time steps in reverse
        T = len(self.inputs_cache)
        
        for t in reversed(range(T)):
            # SOLUTION 16: Gradient from softmax + cross-entropy
            # Convenient property: d(softmax + cross-entropy) = probs - 1(correct)
            dy = self.y_probs[t].copy()
            dy[targets[t]] -= 1
            
            # SOLUTION 17: Gradients for output weights
            # dL/dW = dL/dy * dy/dW = dy * h_t^T (outer product)
            dWy += np.outer(dy, self.h_states[t+1])
            dby += dy
            
            # SOLUTION 18: Gradient w.r.t. hidden state
            # Comes from two sources:
            # 1. From output layer (this time step)
            # 2. From next time step (BPTT)
            dh = np.dot(self.Wy.T, dy) + dh_next
            
            # SOLUTION 19: Gradient w.r.t. output gate
            # From: h_t = o_t ⊙ tanh(C_t)
            # dh = do ⊙ tanh(C_t) + o_t ⊙ d(tanh(C_t))
            # We're computing: do = dh ⊙ tanh(C_t) ⊙ o_t ⊙ (1 - o_t)
            tanh_C = np.tanh(self.C_states[t+1])
            do = dh * tanh_C * self.o_gates[t] * (1 - self.o_gates[t])
            dbo += do
            
            # SOLUTION 20: Gradient w.r.t. cell state
            # From: h_t = o_t ⊙ tanh(C_t)
            # Also include gradient from next time step
            dC = dh * self.o_gates[t] * (1 - tanh_C**2) + dC_next
            
            # SOLUTION 21: Gradient w.r.t. cell candidate
            # From: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            # dC = i_t ⊙ dC̃_t
            dC_tilde = dC * self.i_gates[t] * (1 - self.C_tilde_cache[t]**2)
            dbc += dC_tilde
            
            # SOLUTION 22: Gradient w.r.t. input gate
            # From: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            di = dC * self.C_tilde_cache[t] * self.i_gates[t] * (1 - self.i_gates[t])
            dbi += di
            
            # SOLUTION 23: Gradient w.r.t. forget gate
            # From: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
            df = dC * self.C_states[t] * self.f_gates[t] * (1 - self.f_gates[t])
            dbf += df
            
            # SOLUTION 24: Gradient w.r.t. previous cell state
            # THIS IS THE KEY! Notice it's multiplicative by f_t (usually ~1)
            # So gradients flow almost unchanged - solving vanishing gradients!
            dC_next = dC * self.f_gates[t]
            
            # SOLUTION 25: Gradients for weight matrices
            concat = np.concatenate([self.h_states[t], self.inputs_cache[t]])
            dWo += np.outer(do, concat)
            dWc += np.outer(dC_tilde, concat)
            dWi += np.outer(di, concat)
            dWf += np.outer(df, concat)
            
            # SOLUTION 26: Gradient w.r.t. previous hidden state
            # Comes from all 4 gates (they all use h_{t-1})
            d_concat = (np.dot(self.Wf.T, df) +
                       np.dot(self.Wi.T, di) +
                       np.dot(self.Wc.T, dC_tilde) +
                       np.dot(self.Wo.T, do))
            dh_next = d_concat[:self.hidden_size]
        
        # SOLUTION 27: Gradient clipping (prevents explosion)
        # Clip all gradients to [-5, 5] range
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
        """SOLUTION 28: Update all weights using gradients."""
        self.Wf -= learning_rate * self.dWf
        self.bf -= learning_rate * self.dbf
        self.Wi -= learning_rate * self.dWi
        self.bi -= learning_rate * self.dbi
        self.Wc -= learning_rate * self.dWc
        self.bc -= learning_rate * self.dbc
        self.Wo -= learning_rate * self.dWo
        self.bo -= learning_rate * self.dbo
        self.Wy -= learning_rate * self.dWy
        self.by -= learning_rate * self.dby


# ============================================================================
# Testing
# ============================================================================

def test_lstm():
    """Test the complete LSTM implementation."""
    print("Testing LSTM implementation...")
    print("=" * 60)
    
    # Simple task: learn to repeat "hello"
    text = "hellohellohellohellohello"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Vocabulary: {chars}")
    print(f"Text length: {len(text)}")
    
    vocab_size = len(chars)
    hidden_size = 16
    
    # Create LSTM
    lstm = LSTMFromScratch(vocab_size, hidden_size, vocab_size)
    print(f"\nLSTM created: {vocab_size} inputs, {hidden_size} hidden, {vocab_size} outputs")
    
    # Training
    seq_length = 5
    learning_rate = 0.01
    num_iterations = 200
    
    h = np.zeros(hidden_size)
    C = np.zeros(hidden_size)
    
    losses = []
    
    print(f"\nTraining for {num_iterations} iterations...")
    print("-" * 60)
    
    for iteration in range(num_iterations):
        # Random sequence
        start = np.random.randint(0, len(text) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in text[start:start+seq_length]]
        targets = [char_to_idx[ch] for ch in text[start+1:start+seq_length+1]]
        
        # Forward pass
        loss = lstm.forward(inputs, targets, h, C)
        losses.append(loss)
        
        # Backward pass
        dh, dC = lstm.backward()
        
        # Update weights
        lstm.update_weights(learning_rate)
        
        # Update states (detach from computation graph)
        h = lstm.h_states[-1].copy()
        C = lstm.C_states[-1].copy()
        
        if iteration % 50 == 0:
            smooth_loss = np.mean(losses[-50:]) if len(losses) >= 50 else loss
            print(f"Iteration {iteration:3d}: Loss = {smooth_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    if losses[-1] < losses[0] * 0.5:
        print("\n🎉 Success! Loss decreased significantly.")
        print("Your LSTM is learning!")
    else:
        print("\n⚠️ Loss didn't decrease much. Check your implementation.")


if __name__ == "__main__":
    print(__doc__)
    test_lstm()
