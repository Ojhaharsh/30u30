"""
Solution 1: Dropout Implementation
===================================

Complete solution with explanations for each step.
"""

import numpy as np
from typing import Tuple


class Dropout:
    """
    Dropout layer with inverted dropout scaling.
    
    Key Insight: We use "inverted dropout" - scale during training,
    not during inference. This makes inference faster!
    """
    
    def __init__(self, rate: float = 0.5):
        """
        SOLUTION 1: Initialize dropout
        
        Args:
            rate: Probability of DROPPING a neuron (not keeping)
            
        Why store rate? We need it in forward pass to:
        1. Create the random mask
        2. Scale the kept neurons by 1/(1-rate)
        """
        self.rate = rate
        self.mask = None  # Store for backward pass
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        SOLUTION 2: Forward pass with dropout
        
        During Training:
        1. Generate random mask (same shape as x)
        2. Zero out random neurons (where mask < rate)
        3. Scale remaining by 1/(1-rate)
        
        During Inference:
        - Just return x unchanged (no dropout!)
        
        Why scale by 1/(1-rate)?
        - If we drop 50% of neurons, the remaining sum is half
        - We scale by 2 to maintain the expected sum
        - This way, inference output matches training expected output
        """
        # Inference mode: no dropout
        if not training:
            return x
        
        # No dropout if rate is 0
        if self.rate == 0:
            return x
        
        # SOLUTION 2a: Generate binary mask
        # np.random.rand gives uniform [0, 1)
        # Keep neurons where random > rate
        self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
        
        # SOLUTION 2b: Apply mask and scale
        # Multiply by mask: zeros out dropped neurons
        # Divide by (1-rate): scales up remaining neurons
        return (x * self.mask) / (1 - self.rate)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        SOLUTION 3: Backward pass
        
        Gradient only flows through kept neurons!
        Same mask, same scaling.
        
        If a neuron was dropped (mask=0), gradient is 0.
        If a neuron was kept (mask=1), gradient is scaled by 1/(1-rate).
        """
        return (dout * self.mask) / (1 - self.rate)


class DropoutRNN:
    """
    SOLUTION 4: RNN with proper dropout placement
    
    Dropout placement in RNNs (from the paper):
    - Apply dropout to NON-RECURRENT connections only
    - Input → Hidden: YES dropout
    - Hidden → Hidden: NO dropout (would disrupt memory)
    - Hidden → Output: YES dropout
    
    Why not on recurrent connections?
    - Hidden state carries information across time
    - Dropping bits of memory randomly would corrupt sequences
    - Like forgetting random words in a sentence!
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.5):
        """
        SOLUTION 5: Initialize RNN with dropout layers
        
        Architecture:
            Input → Dropout → Hidden → (no dropout on recurrence) → Output → Dropout
        """
        self.hidden_size = hidden_size
        
        # Weights (Xavier initialization)
        scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (2 * hidden_size))
        scale_ho = np.sqrt(2.0 / (hidden_size + output_size))
        
        self.W_xh = np.random.randn(hidden_size, input_size) * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.W_ho = np.random.randn(output_size, hidden_size) * scale_ho
        self.b_h = np.zeros(hidden_size)
        self.b_o = np.zeros(output_size)
        
        # SOLUTION 5a: Dropout layers
        # Input dropout: applied to input before hidden layer
        # Output dropout: applied to hidden before output layer
        self.input_dropout = Dropout(dropout_rate)
        self.output_dropout = Dropout(dropout_rate)
    
    def forward(self, x_sequence: np.ndarray, training: bool = True) -> Tuple[np.ndarray, list]:
        """
        SOLUTION 6: Forward pass with dropout
        
        Args:
            x_sequence: Input sequence (seq_len, batch_size, input_size)
            training: Whether to apply dropout
            
        Returns:
            output: Final output
            hidden_states: List of hidden states (for backprop)
        """
        seq_len, batch_size, _ = x_sequence.shape
        h = np.zeros((batch_size, self.hidden_size))
        hidden_states = [h]
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            
            # SOLUTION 6a: Apply input dropout
            x_dropped = self.input_dropout.forward(x_t, training)
            
            # SOLUTION 6b: Compute hidden state
            # Note: No dropout on W_hh @ h (recurrent connection)!
            h = np.tanh(
                self.W_xh @ x_dropped.T + 
                self.W_hh @ h.T + 
                self.b_h.reshape(-1, 1)
            ).T
            
            hidden_states.append(h)
        
        # SOLUTION 6c: Apply output dropout before final projection
        h_dropped = self.output_dropout.forward(h, training)
        output = (self.W_ho @ h_dropped.T + self.b_o.reshape(-1, 1)).T
        
        return output, hidden_states


def test_dropout_basic():
    """Test basic dropout functionality."""
    print("Testing basic dropout...")
    
    dropout = Dropout(rate=0.5)
    np.random.seed(42)
    
    x = np.ones((10, 100))
    
    # Training mode
    y_train = dropout.forward(x, training=True)
    dropped = np.sum(y_train == 0)
    print(f"  Training: {dropped} out of 1000 elements dropped (~50% expected)")
    
    # Inference mode
    y_test = dropout.forward(x, training=False)
    assert np.allclose(y_test, x), "Inference should return unchanged input"
    print("  Inference: No dropout applied [ok]")
    
    # Test expected value
    np.random.seed(None)
    sums = []
    for _ in range(1000):
        y = dropout.forward(x, training=True)
        sums.append(np.sum(y))
    
    expected_sum = np.sum(x)  # Should be close due to scaling
    actual_mean = np.mean(sums)
    print(f"  Expected sum: {expected_sum}, Actual mean: {actual_mean:.1f}")
    assert abs(actual_mean - expected_sum) / expected_sum < 0.1, "Scaling incorrect"
    
    print("[ok] Basic dropout tests passed!\n")


def test_rnn_dropout():
    """Test RNN with dropout."""
    print("Testing RNN dropout...")
    
    rnn = DropoutRNN(input_size=5, hidden_size=10, output_size=3, dropout_rate=0.3)
    
    # Create sequence
    seq = np.random.randn(4, 2, 5)  # (seq_len, batch, input)
    
    # Training mode
    out_train, _ = rnn.forward(seq, training=True)
    assert out_train.shape == (2, 3), f"Wrong shape: {out_train.shape}"
    print(f"  Training output shape: {out_train.shape} [ok]")
    
    # Inference mode - should be deterministic
    out1, _ = rnn.forward(seq, training=False)
    out2, _ = rnn.forward(seq, training=False)
    assert np.allclose(out1, out2), "Inference should be deterministic"
    print("  Inference is deterministic [ok]")
    
    print("[ok] RNN dropout tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Solution 1: Dropout Implementation")
    print("="*60 + "\n")
    
    test_dropout_basic()
    test_rnn_dropout()
    
    print("All tests passed!")
