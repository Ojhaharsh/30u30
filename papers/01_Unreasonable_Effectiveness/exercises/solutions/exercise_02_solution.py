"""
Solution to Exercise 2: Temperature Sampling
=============================================

Understanding how temperature controls generation quality.
"""

import numpy as np
import sys
sys.path.append('..')

from implementation import CharRNN


def softmax_with_temperature(logits, temperature=1.0):
    """Apply softmax with temperature scaling."""
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Numerical stability: subtract max
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    
    # Compute probabilities
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities


def generate_with_temperature(rnn, idx_to_char, seed_idx, length, temperature=1.0):
    """Generate text with specific temperature."""
    h = np.zeros(rnn.hidden_size)
    generated = []
    idx = seed_idx
    
    for _ in range(length):
        # One-hot encode input
        x = np.zeros(rnn.vocab_size)
        x[idx] = 1
        
        # Forward pass (single step)
        h = np.tanh(np.dot(rnn.Wxh, x) + np.dot(rnn.Whh, h) + rnn.bh)
        y = np.dot(rnn.Why, h) + rnn.by
        
        # Apply temperature-scaled softmax
        probs = softmax_with_temperature(y, temperature)
        
        # Sample next character
        idx = np.random.choice(range(len(probs)), p=probs)
        generated.append(idx_to_char[idx])
    
    return ''.join(generated)


# ============================================================================
# ANALYSIS
# ============================================================================

"""
Temperature Sampling Analysis
==============================

1. LOW TEMPERATURE (0.2)
   Observations:
   - Text is very repetitive
   - Model picks the most likely character every time
   - Output: "the the the the the"
   - Coherent but boring
   
   Mathematical reason:
   - Low T makes softmax "sharper"
   - Probability mass concentrates on top choice
   - exp(x/0.2) amplifies differences
   
   When to use: When you want safe, predictable output

2. MEDIUM TEMPERATURE (1.0)
   Observations:
   - Balanced between creativity and coherence
   - Text is readable and interesting
   - Occasional spelling errors
   - Good sentence structure
   
   Mathematical reason:
   - T=1.0 is the "natural" softmax (no scaling)
   - Probabilities reflect raw model confidence
   - Allows some randomness, maintains structure
   
   When to use: General purpose, default choice

3. HIGH TEMPERATURE (2.0)
   Observations:
   - Very creative but often nonsensical
   - Made-up words
   - Unusual punctuation
   - Readable but doesn't make sense
   
   Mathematical reason:
   - High T "flattens" softmax distribution
   - exp(x/2.0) reduces differences
   - Almost uniform sampling
   
   When to use: When exploring creative possibilities

4. SWEET SPOT
   For most text datasets: 0.7 - 0.8
   - Coherent structure
   - Some creativity
   - Rare mistakes
   - Interesting but readable

5. MATHEMATICAL INSIGHT
   
   Softmax with temperature:
   p_i = exp(x_i/T) / sum(exp(x_j/T))
   
   Effects:
   - T → 0: argmax sampling (always pick highest)
   - T = 1: standard softmax
   - T → ∞: uniform sampling (all choices equal)
   
   Example with logits [2.0, 1.0, 0.5]:
   
   T = 0.5:  [0.72, 0.22, 0.06]  (very peaked)
   T = 1.0:  [0.59, 0.24, 0.17]  (balanced)
   T = 2.0:  [0.44, 0.31, 0.25]  (almost uniform)
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("This solution demonstrates temperature sampling.")
    print("=" * 60)
    
    print("\nKey insights:")
    print("  - Temperature controls randomness")
    print("  - Low T = repetitive, high T = chaotic")
    print("  - Sweet spot: 0.7-0.8 for most cases")
    print("  - Mathematically: T scales softmax logits")
