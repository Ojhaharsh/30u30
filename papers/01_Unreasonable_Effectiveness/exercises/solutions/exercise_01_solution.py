"""
Solution to Exercise 1: Build a Character-Level RNN from Scratch
=================================================================

This is a complete, working implementation of a character-level RNN.
Compare your solution with this one!
"""

import numpy as np


class CharRNN:
    """
    A minimal character-level Recurrent Neural Network.
    
    This implementation uses pure NumPy and includes:
    - Forward pass (processing sequences)
    - Backward pass (BPTT for computing gradients)
    - Adagrad optimizer
    - Temperature-based sampling
    """
    
    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.01):
        """
        Initialize the RNN with random weights.
        
        Args:
            vocab_size: Number of unique characters
            hidden_size: Dimension of hidden state vector
            seq_length: Number of time steps to unroll
            learning_rate: Step size for gradient descent
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # (Xavier initialization would be better, but this works)
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
        # Adagrad memory (for adaptive learning rates)
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
    def forward(self, inputs, targets, h_prev):
        """
        Forward pass through the RNN.
        
        Processes a sequence of characters and computes:
        - Hidden states at each time step
        - Output probabilities
        - Total loss
        
        Args:
            inputs: List of character indices (length = seq_length)
            targets: List of target character indices (what we want to predict)
            h_prev: Previous hidden state from last sequence (hidden_size x 1)
            
        Returns:
            loss: Total cross-entropy loss for the sequence
            h_last: Final hidden state (to pass to next sequence)
            cache: Intermediate values needed for backward pass
        """
        # Storage for intermediate values
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)  # Initialize with previous hidden state
        loss = 0
        
        # Process each character in the sequence
        for t in range(len(inputs)):
            # 1. Convert character index to one-hot vector
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # 2. Compute new hidden state
            #    h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            #    This combines current input with previous hidden state
            hs[t] = np.tanh(
                self.Wxh @ xs[t] +      # Current input contribution
                self.Whh @ hs[t-1] +    # Previous memory contribution
                self.bh                  # Bias
            )
            
            # 3. Compute unnormalized output scores (logits)
            ys[t] = self.Why @ hs[t] + self.by
            
            # 4. Convert to probabilities via softmax
            #    Using numerically stable softmax: subtract max before exp
            exp_scores = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_scores / np.sum(exp_scores)
            
            # 5. Accumulate cross-entropy loss
            #    Loss = -log(probability of correct character)
            loss += -np.log(ps[t][targets[t], 0])
        
        return loss, hs[len(inputs)-1], (xs, hs, ys, ps)
    
    def backward(self, inputs, targets, cache):
        """
        Backward pass: compute gradients via Backpropagation Through Time (BPTT).
        
        Starting from the loss, we compute gradients for all parameters
        by applying the chain rule backwards through time.
        
        Args:
            inputs: List of character indices
            targets: List of target character indices
            cache: Intermediate values from forward pass (xs, hs, ys, ps)
            
        Returns:
            Dictionary of gradients for all parameters
        """
        xs, hs, ys, ps = cache
        
        # Initialize gradients to zero
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # This will accumulate gradient flowing back from future time steps
        dh_next = np.zeros_like(hs[0])
        
        # Process sequence in reverse (backprop through time)
        for t in reversed(range(len(inputs))):
            # Gradient of loss w.r.t. output scores
            # For softmax + cross-entropy, this simplifies to: p - 1(correct)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            # Backprop to output layer weights
            dWhy += dy @ hs[t].T  # Accumulate gradient
            dby += dy
            
            # Backprop to hidden state
            # Gradient comes from two sources:
            # 1. Directly from output layer (Why.T @ dy)
            # 2. From future time step (dh_next)
            dh = self.Why.T @ dy + dh_next
            
            # Backprop through tanh nonlinearity
            # d/dx tanh(x) = 1 - tanh(x)^2
            dh_raw = (1 - hs[t] ** 2) * dh
            
            # Backprop to hidden layer weights and biases
            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t-1].T
            
            # Gradient to pass to previous time step
            dh_next = self.Whh.T @ dh_raw
        
        # Clip gradients to prevent explosion
        # This is crucial for RNN training stability!
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return {
            'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy,
            'bh': dbh, 'by': dby
        }
    
    def update_weights(self, grads):
        """
        Update weights using Adagrad optimizer.
        
        Adagrad adapts the learning rate for each parameter based on
        historical gradients (parameters with larger gradients get smaller updates).
        
        Args:
            grads: Dictionary of gradients from backward pass
        """
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        grads_list = [grads['Wxh'], grads['Whh'], grads['Why'], grads['bh'], grads['by']]
        memories = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        
        for param, grad, mem in zip(params, grads_list, memories):
            # Accumulate squared gradients
            mem += grad * grad
            
            # Update with adaptive learning rate
            # lr / sqrt(sum of squared gradients + epsilon)
            param -= self.learning_rate * grad / np.sqrt(mem + 1e-8)
    
    def sample(self, h, seed_idx, n, temperature=1.0):
        """
        Generate text by sampling from the model.
        
        Args:
            h: Initial hidden state
            seed_idx: Index of starting character
            n: Number of characters to generate
            temperature: Controls randomness (higher = more random)
            
        Returns:
            List of generated character indices
        """
        # Start with seed character
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        for t in range(n):
            # Forward pass for one step
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            
            # Apply temperature scaling
            y = y / temperature
            
            # Softmax to get probabilities
            p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
            
            # Sample from the distribution
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # Use sampled character as input for next step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)
        
        return indices


def train_on_simple_data():
    """
    Train the RNN on simple repeating pattern.
    
    This is a good test to make sure the implementation works!
    """
    print("Training on simple data: 'abcabcabc...'")
    print("=" * 60)
    
    # Simple training data
    data = "abcabcabc" * 20  # Repeat pattern
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    print(f"Vocabulary: {chars}")
    print(f"Data length: {len(data)} characters\n")
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Convert data to indices
    data_indices = [char_to_idx[ch] for ch in data]
    
    # Initialize RNN
    rnn = CharRNN(vocab_size=vocab_size, hidden_size=20, seq_length=10, learning_rate=0.1)
    
    # Training loop
    p = 0  # Data pointer
    h_prev = np.zeros((rnn.hidden_size, 1))
    smooth_loss = -np.log(1.0/vocab_size) * rnn.seq_length
    
    for iteration in range(500):
        # Reset at end of data
        if p + rnn.seq_length + 1 >= len(data_indices) or iteration == 0:
            h_prev = np.zeros((rnn.hidden_size, 1))
            p = 0
        
        # Get sequence
        inputs = data_indices[p:p+rnn.seq_length]
        targets = data_indices[p+1:p+rnn.seq_length+1]
        
        # Forward pass
        loss, h_prev, cache = rnn.forward(inputs, targets, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
        # Backward pass
        grads = rnn.backward(inputs, targets, cache)
        
        # Update weights
        rnn.update_weights(grads)
        
        # Move pointer
        p += rnn.seq_length
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {smooth_loss:.4f}")
            
            # Generate sample
            h = np.zeros((rnn.hidden_size, 1))
            sample_indices = rnn.sample(h, char_to_idx['a'], 30, temperature=0.5)
            txt = ''.join(idx_to_char[i] for i in sample_indices)
            print(f"  Generated: '{txt}'\n")
    
    print("=" * 60)
    print("Training complete! ✅")
    print("\nFinal test - Generate with different temperatures:")
    
    for temp in [0.2, 0.5, 1.0, 2.0]:
        h = np.zeros((rnn.hidden_size, 1))
        sample_indices = rnn.sample(h, char_to_idx['a'], 50, temperature=temp)
        txt = ''.join(idx_to_char[i] for i in sample_indices)
        print(f"\nTemperature {temp}: '{txt}'")


if __name__ == "__main__":
    train_on_simple_data()
