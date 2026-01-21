"""
LSTM (Long Short-Term Memory) Implementation in NumPy
=====================================================

A complete, educational implementation of LSTM networks.

This is more complex than vanilla RNNs (Day 1) because we have:
- 4 weight matrices (forget, input, cell, output gates)
- Cell state in addition to hidden state
- More complex backpropagation

But the extra complexity solves the vanishing gradient problem!

Author: 30u30 Project
License: MIT
"""

import numpy as np
import pickle


def sigmoid(x):
    """
    Sigmoid activation function.
    
    Maps any value to range (0, 1).
    Used for gates because we want values between "fully open" (1) and "fully closed" (0).
    
    Args:
        x: Input array
        
    Returns:
        Array with values between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """
    Softmax activation (numerically stable version).
    
    Converts logits to probabilities that sum to 1.
    
    Args:
        x: Input array of logits
        
    Returns:
        Probability distribution
    """
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


class LSTM:
    """
    Character-level LSTM Language Model.
    
    This LSTM learns to predict the next character in a sequence.
    It's more powerful than vanilla RNN because it can remember information
    for longer periods thanks to the cell state and gating mechanism.
    
    Architecture:
        - Input: One-hot encoded characters
        - Hidden layer: LSTM with forget, input, and output gates
        - Output: Softmax over character vocabulary
    """
    
    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.001):
        """
        Initialize LSTM with random weights.
        
        Args:
            vocab_size: Number of unique characters in vocabulary
            hidden_size: Dimension of hidden state and cell state
            seq_length: Number of time steps to unroll during training
            learning_rate: Step size for gradient descent (smaller than RNN!)
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # LSTM has 4 sets of weights (forget, input, cell, output)
        # Each transforms [hidden_state, input] → gate activations
        # Using Xavier initialization: scale by sqrt(2 / (fan_in))
        scale = np.sqrt(2.0 / (hidden_size + vocab_size))
        
        # Forget gate: decides what to forget from cell state
        self.Wf = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bf = np.ones((hidden_size, 1))  # Initialize to 1 (default: remember)
        
        # Input gate: decides what new information to store
        self.Wi = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        
        # Cell candidate: computes candidate values to add to cell state
        self.Wc = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        
        # Output gate: decides what to output from cell state
        self.Wo = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        
        # Output layer: projects hidden state to vocabulary
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.by = np.zeros((vocab_size, 1))
        
        # Adagrad memory for adaptive learning rates
        # We track sum of squared gradients for each parameter
        self.mWf, self.mWi, self.mWc, self.mWo = [np.zeros_like(W) for W in [self.Wf, self.Wi, self.Wc, self.Wo]]
        self.mbf, self.mbi, self.mbc, self.mbo = [np.zeros_like(b) for b in [self.bf, self.bi, self.bc, self.bo]]
        self.mWhy = np.zeros_like(self.Why)
        self.mby = np.zeros_like(self.by)
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        Forward pass through the LSTM for a sequence of inputs.
        
        This processes seq_length characters, computing:
        1. Gate activations (forget, input, output)
        2. Cell state updates
        3. Hidden state updates
        4. Output predictions
        5. Loss
        
        Args:
            inputs: List of character indices (length = seq_length)
            targets: List of target character indices (what we want to predict)
            h_prev: Previous hidden state (hidden_size × 1)
            C_prev: Previous cell state (hidden_size × 1)
            
        Returns:
            loss: Total cross-entropy loss for this sequence
            h_last: Final hidden state (to pass to next sequence)
            C_last: Final cell state (to pass to next sequence)
            cache: All intermediate values needed for backward pass
        """
        # Storage for intermediate values (needed for backprop)
        xs, hs, Cs, ys, ps = {}, {}, {}, {}, {}
        fs, i_s, c_cands, os, zs = {}, {}, {}, {}, {}  # Gate activations
        
        # Initialize with previous states
        hs[-1] = np.copy(h_prev)
        Cs[-1] = np.copy(C_prev)
        
        loss = 0
        
        # Forward pass through time
        for t in range(len(inputs)):
            # === Step 1: Prepare input ===
            # Convert character index to one-hot vector
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # Concatenate previous hidden state with current input
            # This is what all gates will use to make decisions
            zs[t] = np.vstack([hs[t-1], xs[t]])  # Shape: (hidden_size + vocab_size, 1)
            
            # === Step 2: Forget gate ===
            # "What should we forget from the cell state?"
            # f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
            # Output: values between 0 (forget everything) and 1 (keep everything)
            fs[t] = sigmoid(self.Wf @ zs[t] + self.bf)
            
            # === Step 3: Input gate + Cell candidate ===
            # "What new information should we add to cell state?"
            # i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)    # How much to let in
            # ~C_t = tanh(W_c · [h_{t-1}, x_t] + b_c)      # Candidate values
            i_s[t] = sigmoid(self.Wi @ zs[t] + self.bi)
            c_cands[t] = np.tanh(self.Wc @ zs[t] + self.bc)
            
            # === Step 4: Cell state update ===
            # "Update long-term memory"
            # C_t = f_t ⊙ C_{t-1} + i_t ⊙ ~C_t
            # This is the key: we can preserve information by setting f_t ≈ 1!
            Cs[t] = fs[t] * Cs[t-1] + i_s[t] * c_cands[t]
            
            # === Step 5: Output gate ===
            # "What should we output based on cell state?"
            # o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
            os[t] = sigmoid(self.Wo @ zs[t] + self.bo)
            
            # === Step 6: Hidden state ===
            # "Compute hidden state from cell state"
            # h_t = o_t ⊙ tanh(C_t)
            hs[t] = os[t] * np.tanh(Cs[t])
            
            # === Step 7: Output layer ===
            # Project hidden state to vocabulary size
            ys[t] = self.Why @ hs[t] + self.by
            
            # === Step 8: Softmax ===
            # Convert to probability distribution
            ps[t] = softmax(ys[t])
            
            # === Step 9: Loss ===
            # Cross-entropy loss: -log(probability of correct character)
            loss += -np.log(ps[t][targets[t], 0])
        
        # Cache everything for backward pass
        cache = {
            'xs': xs, 'hs': hs, 'Cs': Cs, 'ys': ys, 'ps': ps,
            'fs': fs, 'i_s': i_s, 'c_cands': c_cands, 'os': os, 'zs': zs
        }
        
        return loss, hs[len(inputs)-1], Cs[len(inputs)-1], cache
    
    def backward(self, inputs, targets, cache):
        """
        Backward pass: compute gradients via Backpropagation Through Time.
        
        This is more complex than vanilla RNN because we have:
        1. Cell state gradients (in addition to hidden state)
        2. Four gates to backprop through
        3. More careful gradient flow management
        
        The good news: The cell state allows gradients to flow more easily!
        
        Args:
            inputs: List of character indices
            targets: List of target character indices
            cache: Intermediate values from forward pass
            
        Returns:
            Dictionary of gradients for all parameters
        """
        xs = cache['xs']
        hs = cache['hs']
        Cs = cache['Cs']
        ps = cache['ps']
        fs = cache['fs']
        i_s = cache['i_s']
        c_cands = cache['c_cands']
        os = cache['os']
        zs = cache['zs']
        
        # Initialize gradients to zero
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dWhy = np.zeros_like(self.Why)
        
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dby = np.zeros_like(self.by)
        
        # These accumulate gradients flowing backward through time
        dh_next = np.zeros_like(hs[0])
        dC_next = np.zeros_like(Cs[0])
        
        # Backpropagate through time (in reverse order)
        for t in reversed(range(len(inputs))):
            # === Output layer gradients ===
            # Gradient of loss w.r.t. output scores (softmax + cross-entropy)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # This is the elegant softmax derivative!
            
            dWhy += dy @ hs[t].T
            dby += dy
            
            # === Backprop to hidden state ===
            # Gradient comes from:
            # 1. Output layer (Why.T @ dy)
            # 2. Next time step (dh_next)
            dh = self.Why.T @ dy + dh_next
            
            # === Output gate gradients ===
            # h_t = o_t ⊙ tanh(C_t)
            # ∂L/∂o_t = ∂L/∂h_t ⊙ tanh(C_t)
            do = dh * np.tanh(Cs[t])
            # Backprop through sigmoid: σ'(x) = σ(x) * (1 - σ(x))
            do_input = do * os[t] * (1 - os[t])
            
            # === Cell state gradients ===
            # Gradient comes from:
            # 1. Hidden state: dh * o_t * tanh'(C_t)
            # 2. Next time step: dC_next
            dC = dh * os[t] * (1 - np.tanh(Cs[t])**2) + dC_next
            
            # === Forget gate gradients ===
            # C_t = f_t ⊙ C_{t-1} + ...
            # ∂L/∂f_t = ∂L/∂C_t ⊙ C_{t-1}
            df = dC * Cs[t-1]
            df_input = df * fs[t] * (1 - fs[t])
            
            # === Input gate gradients ===
            # C_t = ... + i_t ⊙ ~C_t
            # ∂L/∂i_t = ∂L/∂C_t ⊙ ~C_t
            di = dC * c_cands[t]
            di_input = di * i_s[t] * (1 - i_s[t])
            
            # === Cell candidate gradients ===
            # C_t = ... + i_t ⊙ ~C_t
            # ∂L/∂~C_t = ∂L/∂C_t ⊙ i_t
            dc_cand = dC * i_s[t]
            # Backprop through tanh: tanh'(x) = 1 - tanh(x)^2
            dc_cand_input = dc_cand * (1 - c_cands[t]**2)
            
            # === Weight gradients (all gates) ===
            # Each gate transforms z_t = [h_{t-1}, x_t], so:
            # ∂L/∂W = ∂L/∂(W·z) @ z^T
            dWf += df_input @ zs[t].T
            dWi += di_input @ zs[t].T
            dWc += dc_cand_input @ zs[t].T
            dWo += do_input @ zs[t].T
            
            # === Bias gradients ===
            dbf += df_input
            dbi += di_input
            dbc += dc_cand_input
            dbo += do_input
            
            # === Backprop to [h_{t-1}, x_t] ===
            dz = (self.Wf.T @ df_input + 
                  self.Wi.T @ di_input +
                  self.Wc.T @ dc_cand_input +
                  self.Wo.T @ do_input)
            
            # Split into gradients for h_{t-1} and x_t
            dh_next = dz[:self.hidden_size]
            # dx = dz[self.hidden_size:]  # We don't need this
            
            # === Cell state gradient to previous time step ===
            # This is KEY: gradient flows through via forget gate!
            # dC_{t-1} = ∂L/∂C_t ⊙ f_t
            # If f_t ≈ 1, gradient flows easily!
            dC_next = dC * fs[t]
        
        # Clip gradients to prevent explosion
        # (LSTMs reduce vanishing but can still explode!)
        for grad in [dWf, dWi, dWc, dWo, dWhy, dbf, dbi, dbc, dbo, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        return {
            'Wf': dWf, 'Wi': dWi, 'Wc': dWc, 'Wo': dWo, 'Why': dWhy,
            'bf': dbf, 'bi': dbi, 'bc': dbc, 'bo': dbo, 'by': dby
        }
    
    def update_weights(self, grads):
        """
        Update weights using Adagrad optimizer.
        
        Adagrad adapts learning rate for each parameter based on
        historical gradients. Parameters with large gradients get
        smaller effective learning rates.
        
        Args:
            grads: Dictionary of gradients from backward pass
        """
        params = {
            'Wf': self.Wf, 'Wi': self.Wi, 'Wc': self.Wc, 'Wo': self.Wo, 'Why': self.Why,
            'bf': self.bf, 'bi': self.bi, 'bc': self.bc, 'bo': self.bo, 'by': self.by
        }
        
        memories = {
            'Wf': self.mWf, 'Wi': self.mWi, 'Wc': self.mWc, 'Wo': self.mWo, 'Why': self.mWhy,
            'bf': self.mbf, 'bi': self.mbi, 'bc': self.mbc, 'bo': self.mbo, 'by': self.mby
        }
        
        for key in params:
            # Accumulate squared gradients
            memories[key] += grads[key] * grads[key]
            
            # Update with adaptive learning rate
            # learning_rate / sqrt(sum of squared gradients + epsilon)
            params[key] -= self.learning_rate * grads[key] / np.sqrt(memories[key] + 1e-8)
    
    def sample(self, h, C, seed_idx, n, temperature=1.0):
        """
        Generate text by sampling from the LSTM.
        
        Args:
            h: Initial hidden state (hidden_size × 1)
            C: Initial cell state (hidden_size × 1)
            seed_idx: Index of starting character
            n: Number of characters to generate
            temperature: Controls randomness (higher = more random)
            
        Returns:
            List of generated character indices
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        for t in range(n):
            # Concatenate h and x
            z = np.vstack([h, x])
            
            # Forward pass through LSTM cell
            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            c_cand = np.tanh(self.Wc @ z + self.bc)
            C = f * C + i * c_cand
            o = sigmoid(self.Wo @ z + self.bo)
            h = o * np.tanh(C)
            
            # Output layer
            y = self.Why @ h + self.by
            
            # Apply temperature
            y = y / temperature
            
            # Softmax
            p = softmax(y)
            
            # Sample from distribution
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # Update input for next step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)
        
        return indices
    
    def save(self, filepath):
        """Save model parameters to file."""
        params = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'learning_rate': self.learning_rate,
            'Wf': self.Wf, 'Wi': self.Wi, 'Wc': self.Wc, 'Wo': self.Wo, 'Why': self.Why,
            'bf': self.bf, 'bi': self.bi, 'bc': self.bc, 'bo': self.bo, 'by': self.by
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model parameters from file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        model = cls(params['vocab_size'], params['hidden_size'], 
                   params['seq_length'], params['learning_rate'])
        
        model.Wf = params['Wf']
        model.Wi = params['Wi']
        model.Wc = params['Wc']
        model.Wo = params['Wo']
        model.Why = params['Why']
        model.bf = params['bf']
        model.bi = params['bi']
        model.bc = params['bc']
        model.bo = params['bo']
        model.by = params['by']
        
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    print("LSTM Implementation")
    print("=" * 60)
    print("\nThis file contains the LSTM class.")
    print("Use train_minimal.py to train a model.")
    print("Use notebook.ipynb for interactive learning.")
    print("\nKey differences from vanilla RNN:")
    print("  - Cell state (C) in addition to hidden state (h)")
    print("  - Four weight matrices (forget, input, cell, output)")
    print("  - Gates control information flow")
    print("  - Gradients flow more easily through cell state")
