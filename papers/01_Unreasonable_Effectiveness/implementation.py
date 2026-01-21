"""
Minimal Character-Level RNN Implementation
Based on Andrej Karpathy's min-char-rnn.py
Heavily commented for educational purposes

This is a complete, working implementation of a character-level RNN in pure NumPy.
Every line is explained. No magic.
"""

import numpy as np


class CharRNN:
    """
    A minimal character-level Recurrent Neural Network.
    
    Architecture:
    - Input: One-hot encoded characters
    - Hidden layer: RNN with tanh activation
    - Output: Softmax over vocabulary
    
    Parameters:
    -----------
    vocab_size : int
        Number of unique characters in vocabulary
    hidden_size : int
        Size of hidden state (more = more memory, but slower)
    learning_rate : float
        How big of steps to take during learning
    seq_length : int
        How many characters to process before updating weights
    """
    
    def __init__(self, vocab_size, hidden_size=100, learning_rate=1e-1, seq_length=25):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        
        # Initialize weight matrices with small random values
        # Why small? Large values can cause exploding gradients
        # Why random? Breaks symmetry (all neurons would learn the same thing otherwise)
        
        # Wxh: Weight matrix from input (x) to hidden (h)
        # Shape: (hidden_size, vocab_size)
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        
        # Whh: Weight matrix from previous hidden (h) to current hidden (h)
        # Shape: (hidden_size, hidden_size)
        # This is the "memory" connection
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # Why: Weight matrix from hidden (h) to output (y)
        # Shape: (vocab_size, hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # Biases (one per neuron)
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((vocab_size, 1))   # Output bias
        
        # Memory for Adagrad (adaptive learning rate)
        # Adagrad adapts learning rate per parameter based on history
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
    def forward(self, inputs, hprev):
        """
        Forward pass through the RNN.
        
        Parameters:
        -----------
        inputs : list of int
            List of character indices (encoded as integers)
        hprev : numpy array
            Previous hidden state (shape: hidden_size x 1)
            
        Returns:
        --------
        xs : dict
            Input vectors at each time step
        hs : dict
            Hidden states at each time step
        ys : dict
            Output (logits) at each time step
        ps : dict
            Probabilities (after softmax) at each time step
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)  # Initial hidden state
        
        # Process each character in the sequence
        for t, char_idx in enumerate(inputs):
            # 1. Convert character index to one-hot vector
            # Example: if vocab_size=5 and char_idx=2, xs[t] = [0, 0, 1, 0, 0]
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][char_idx] = 1
            
            # 2. Compute new hidden state
            # h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
            #       ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
            #       process input     remember past
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) +   # Input contribution
                np.dot(self.Whh, hs[t-1]) + # Previous state (memory!)
                self.bh                      # Bias
            )
            
            # 3. Compute output scores (logits)
            # y_t = Why @ h_t + by
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            
            # 4. Convert scores to probabilities (softmax)
            # Softmax: e^x / sum(e^x) → all probabilities sum to 1
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
        return xs, hs, ys, ps
    
    def loss(self, ps, targets):
        """
        Compute cross-entropy loss.
        
        Cross-entropy measures how wrong predictions are.
        Lower loss = better predictions
        
        Parameters:
        -----------
        ps : dict
            Predicted probabilities at each time step
        targets : list of int
            True character indices
            
        Returns:
        --------
        loss : float
            Average loss per character
        """
        loss = 0
        for t, target in enumerate(targets):
            # Cross-entropy: -log(predicted probability of correct character)
            # If predicted probability = 1.0 → loss = 0 (perfect!)
            # If predicted probability = 0.01 → loss = -log(0.01) ≈ 4.6 (bad!)
            loss += -np.log(ps[t][target, 0])
        return loss / len(targets)
    
    def backward(self, xs, hs, ps, targets):
        """
        Backward pass: compute gradients via backpropagation through time (BPTT).
        
        This is where the learning happens. We compute:
        - How much each weight contributed to the error
        - Which direction to adjust weights to reduce error
        
        Parameters:
        -----------
        xs, hs, ps : dicts (from forward pass)
        targets : list of int
            True character indices
            
        Returns:
        --------
        gradients : dict
            Gradients for each weight matrix
        """
        # Initialize gradient accumulators
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])  # Gradient flowing back from future
        
        # Backpropagate through time (from end to start)
        for t in reversed(range(len(targets))):
            # 1. Compute gradient of loss with respect to output (dy)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # Softmax gradient: p - 1 for correct class, p for others
            
            # 2. Gradients for output layer weights
            dWhy += np.dot(dy, hs[t].T)  # @ is gradient through matrix multiply
            dby += dy
            
            # 3. Gradient flows back to hidden state
            dh = np.dot(self.Why.T, dy) + dhnext  # From output + from future
            
            # 4. Gradient through tanh
            # tanh'(x) = 1 - tanh²(x)
            dhraw = (1 - hs[t] * hs[t]) * dh
            
            # 5. Gradients for hidden layer weights
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            
            # 6. Gradient flows to previous time step
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # Clip gradients to prevent exploding gradients
        # Without this, gradients can get huge and training explodes
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def update_parameters(self, gradients):
        """
        Update weights using Adagrad.
        
        Adagrad adapts learning rate for each parameter:
        - Parameters with large gradients get small updates
        - Parameters with small gradients get larger updates
        This helps prevent oscillation and speeds up convergence.
        
        Parameters:
        -----------
        gradients : tuple
            (dWxh, dWhh, dWhy, dbh, dby)
        """
        dWxh, dWhh, dWhy, dbh, dby = gradients
        
        # Update memory (sum of squared gradients)
        self.mWxh += dWxh * dWxh
        self.mWhh += dWhh * dWhh
        self.mWhy += dWhy * dWhy
        self.mbh += dbh * dbh
        self.mby += dby * dby
        
        # Update parameters (Adagrad formula)
        # parameter -= learning_rate * gradient / sqrt(memory + eps)
        # eps prevents division by zero
        self.Wxh -= self.learning_rate * dWxh / np.sqrt(self.mWxh + 1e-8)
        self.Whh -= self.learning_rate * dWhh / np.sqrt(self.mWhh + 1e-8)
        self.Why -= self.learning_rate * dWhy / np.sqrt(self.mWhy + 1e-8)
        self.bh -= self.learning_rate * dbh / np.sqrt(self.mbh + 1e-8)
        self.by -= self.learning_rate * dby / np.sqrt(self.mby + 1e-8)
    
    def sample(self, h, seed_idx, n_chars, temperature=1.0):
        """
        Generate text by sampling from the model.
        
        Parameters:
        -----------
        h : numpy array
            Initial hidden state
        seed_idx : int
            Starting character index
        n_chars : int
            Number of characters to generate
        temperature : float
            Controls randomness (lower = more conservative, higher = more random)
            1.0 = use raw probabilities
            0.5 = sharpen (more conservative)
            2.0 = flatten (more random)
            
        Returns:
        --------
        indices : list of int
            Generated character indices
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        for _ in range(n_chars):
            # Forward pass (same as training, but only one step)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            # Apply temperature
            # Temperature controls randomness:
            # - Higher temp → more uniform distribution (more creative/random)
            # - Lower temp → more peaked distribution (more deterministic)
            y = y / temperature
            p = np.exp(y) / np.sum(np.exp(y))
            
            # Sample from probability distribution
            # This introduces randomness (more interesting than always picking max)
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # Use sampled character as input for next step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            
            indices.append(idx)
        
        return indices


def prepare_data(text):
    """
    Convert text to integer indices.
    
    Returns:
    --------
    char_to_idx : dict
        Mapping from character to index
    idx_to_char : dict
        Mapping from index to character
    data : list of int
        Text as list of indices
    """
    chars = sorted(list(set(text)))  # Unique characters
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_idx[ch] for ch in text]
    return char_to_idx, idx_to_char, data


if __name__ == "__main__":
    # Demo: Train on a tiny example
    text = "hello world, this is a minimal RNN implementation. " * 10
    char_to_idx, idx_to_char, data = prepare_data(text)
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Data length: {len(data)} characters")
    print(f"Sample: {text[:50]}...")
    
    # Initialize model
    model = CharRNN(
        vocab_size=len(char_to_idx),
        hidden_size=100,
        learning_rate=1e-1,
        seq_length=25
    )
    
    print("\nTraining...")
    print("(Run train_minimal.py for full training with progress bars)")
