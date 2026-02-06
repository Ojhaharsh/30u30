# Exercises: Build Bahdanau Attention from Scratch

These exercises walk you through building the complete attention mechanism
piece by piece, following the structure of the original paper.

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Additive Attention | 2/5 | 30 min |
| 2 | Bidirectional Encoder | 2/5 | 25 min |
| 3 | Attention Decoder | 3/5 | 40 min |
| 4 | Training Loop | 2/5 | 30 min |
| 5 | Visualization | 1/5 | 20 min |

## Exercise 1: Additive Attention

**The Core Mechanism**

Implement the attention scoring function:
```
score(s, h) = v^T * tanh(W_s * s + W_h * h)
```

You'll learn:
- Why it's called "additive" (we ADD transformed vectors)
- How attention weights are computed
- The role of softmax normalization

## Exercise 2: Bidirectional Encoder

**Bidirectional Encoding**

Build an encoder that processes the source sequence in both directions:
- Forward: left -> right
- Backward: right -> left

You'll learn:
- Why bidirectional context helps attention
- How to handle packed sequences for efficiency
- Combining forward/backward hidden states

## Exercise 3: Attention Decoder

**Attention-Based Decoder**

Implement a decoder that:
1. Computes attention over encoder outputs
2. Creates a context vector
3. Combines context with input for prediction

You'll learn:
- The attention-GRU interaction
- Teacher forcing for training
- Step-by-step vs full sequence decoding

## Exercise 4: Training Loop

**Full Pipeline**

Complete the training loop for the reversal task:
- Loss computation with masking
- Gradient clipping
- Validation and accuracy calculation

Goal: **Achieve >90% accuracy** on sequence reversal.

## Exercise 5: Visualization

**Attention Visualization**

Create attention heatmaps:
- Visualize the reversed diagonal pattern
- Analyze attention entropy
- Debug model behavior visually

## How to Use

1. **Read the exercise file** - Each has detailed instructions
2. **Find the TODO sections** - These are what you implement
3. **Run the tests** - Each file has a test function
4. **Check solutions** - Compare with `solutions/solution_X.py`

## Running Exercises

```bash
# Test your implementation
python exercise_1.py

# After completing all exercises, run full training
python exercise_4.py
```

## Tips

- **Start with Exercise 1** - Attention is the key concept
- **Read the docstrings** - They contain hints
- **Use the tests** - They verify your implementation
- **Check shapes** - Print tensor shapes when debugging
- **Compare with solutions** - But try first.

## Common Issues

### "Shapes don't match"
- Check batch dimension handling
- Verify squeeze/unsqueeze operations
- Print shapes at each step

### "Attention weights don't sum to 1"
- Make sure softmax is on the right dimension
- Check masking implementation

### "Training loss doesn't decrease"
- Verify gradient flow (check .grad attributes)
- Try smaller learning rate
- Check that loss ignores padding

The attention mechanism is a widely used concept in modern deep learning. Understanding it provides a foundation for many subsequent architectures.
