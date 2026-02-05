"""
Exercise 1: Build a Character-Level RNN from Scratch
=====================================================

Your task: Fill in the TODOs to create a working RNN.

This will help you understand:
- How data flows through an RNN
- How gradients are computed via BPTT
- Why gradient clipping matters
"""

import numpy as np

class CharRNN:
    """
    A minimal character-level RNN for learning.
    
    TODO: Implement the missing methods below.
    """
    
    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.01):
        """
        Initialize the RNN.
        
        Args:
            vocab_size: Number of unique characters
            hidden_size: Size of hidden state vector
            seq_length: Number of characters to process at once
            learning_rate: Step size for gradient descent
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # TODO: Initialize weight matrices
        # Hint: Use small random values, e.g., np.random.randn(...) * 0.01
        #
        # You need:
        # - Wxh: Input to hidden (vocab_size x hidden_size)
        # - Whh: Hidden to hidden (hidden_size x hidden_size)
        # - Why: Hidden to output (hidden_size x vocab_size)
        # - bh: Hidden bias (hidden_size x 1)
        # - by: Output bias (vocab_size x 1)
        
        self.Wxh = None  # TODO
        self.Whh = None  # TODO
        self.Why = None  # TODO
        self.bh = None   # TODO
        self.by = None   # TODO
        
    def forward(self, inputs, h_prev):
        """
        Forward pass through the RNN.
        
        Args:
            inputs: List of character indices (length = seq_length)
            h_prev: Previous hidden state (hidden_size x 1)
            
        Returns:
            loss: Cross-entropy loss
            h_last: Final hidden state
            cache: Values needed for backward pass
        """
        
        # Storage for forward pass
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        # TODO: Implement forward pass through time
        # For each time step t:
        #   1. Create one-hot encoding for inputs[t]
        #   2. Compute hidden state: h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
        #   3. Compute output: y_t = Why @ h_t + by
        #   4. Compute probabilities: p_t = softmax(y_t)
        #   5. Accumulate loss: -log(p_t[target_t])
        #
        # Hint: Store xs, hs, ys, ps in dictionaries for backward pass
        
        for t in range(len(inputs)):
            # TODO: Your code here
            pass
        
        return loss, hs[len(inputs)-1], (xs, hs, ys, ps)
    
    def backward(self, inputs, targets, cache):
        """
        Backward pass: compute gradients via BPTT.
        
        Args:
            inputs: List of character indices
            targets: List of target character indices
            cache: Values from forward pass (xs, hs, ys, ps)
            
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        
        xs, hs, ys, ps = cache
        
        # Initialize gradients to zero
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        # TODO: Implement backward pass through time
        # For each time step t (in reverse order):
        #   1. Compute gradient of loss w.r.t. output: dy = p_t (copy), dy[target_t] -= 1
        #   2. Accumulate output layer gradients: dWhy, dby
        #   3. Backpropagate to hidden layer: dh = Why.T @ dy + dh_next
        #   4. Backpropagate through tanh: dh_raw = (1 - h_t^2) * dh
        #   5. Accumulate hidden layer gradients: dWxh, dWhh, dbh
        #   6. Propagate gradient to previous time step: dh_next = Whh.T @ dh_raw
        #   7. Clip gradients to prevent explosion
        #
        # Hint: Use np.clip(grad, -5, 5) for gradient clipping
        
        for t in reversed(range(len(inputs))):
            # TODO: Your code here
            pass
        
        # TODO: Clip gradients
        # for gradient in [dWxh, dWhh, dWhy, dbh, dby]:
        #     np.clip(gradient, -5, 5, out=gradient)
        
        return {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}
    
    def update_weights(self, grads):
        """
        Update weights using gradient descent.
        
        Args:
            grads: Dictionary of gradients
        """
        
        # TODO: Update each parameter
        # param = param - learning_rate * grad
        
        pass
    
    def sample(self, h, seed_idx, n):
        """
        Generate text by sampling from the model.
        
        Args:
            h: Initial hidden state
            seed_idx: Starting character index
            n: Number of characters to generate
            
        Returns:
            indices: List of generated character indices
        """
        
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        # TODO: Implement sampling
        # For each step:
        #   1. Compute h = tanh(Wxh @ x + Whh @ h + bh)
        #   2. Compute y = Why @ h + by
        #   3. Compute p = softmax(y)
        #   4. Sample next character from p
        #   5. Use sampled character as input for next step
        
        for t in range(n):
            # TODO: Your code here
            pass
        
        return indices


def train_rnn():
    """
    Train the RNN on simple data.
    
    Test your implementation here!
    """
    
    # Simple training data
    data = "abcabcabcabcabcabc"
    chars = list(set(data))
    vocab_size = len(chars)
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Convert data to indices
    data_indices = [char_to_idx[ch] for ch in data]
    
    # TODO: Initialize your RNN
    rnn = CharRNN(vocab_size=vocab_size, hidden_size=10, seq_length=3)
    
    # TODO: Training loop
    # For each epoch:
    #   1. Sample a sequence from data
    #   2. Forward pass
    #   3. Backward pass
    #   4. Update weights
    #   5. Print loss
    
    # Hint: Start with just 100 iterations
    
    print("Training complete!")
    
    # TODO: Generate some text
    # h = np.zeros((rnn.hidden_size, 1))
    # indices = rnn.sample(h, seed_idx=0, n=20)
    # text = ''.join([idx_to_char[i] for i in indices])
    # print(f"Generated: {text}")


if __name__ == "__main__":
    print("Exercise 1: Build a Character-Level RNN")
    print("=" * 50)
    print("\nYour task: Fill in the TODOs above.")
    print("\nSteps:")
    print("1. Initialize weight matrices")
    print("2. Implement forward pass")
    print("3. Implement backward pass")
    print("4. Implement weight updates")
    print("5. Implement sampling")
    print("\nGood luck!")
    print("\n" + "=" * 50)
    
    # Uncomment when you're ready to test:
    # train_rnn()
