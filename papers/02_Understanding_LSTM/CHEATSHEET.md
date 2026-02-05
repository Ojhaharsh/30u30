# LSTM Cheatsheet

Quick reference for Understanding LSTM Networks

---

## The Big Idea (30 seconds)

LSTMs solve vanishing gradients by adding a **cell state** that allows gradients to flow through time via addition rather than multiplication. Colah's metaphor: the cell state is a "conveyor belt" — information rides along it unchanged unless the LSTM explicitly modifies it through gates.

---

## Architecture: The 4 Gates

```
Forget Gate:    f_t = σ(W_f·[h_{t-1}, x_t] + b_f)    # 0 = forget, 1 = keep
Input Gate:     i_t = σ(W_i·[h_{t-1}, x_t] + b_i)    # 0 = ignore, 1 = add
Cell Candidate: C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C) # New info to add
Output Gate:    o_t = σ(W_o·[h_{t-1}, x_t] + b_o)    # 0 = hide, 1 = show

Cell State:     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t     # The memory
Hidden State:   h_t = o_t ⊙ tanh(C_t)                 # The output
```

**Key insight**: Cell state updates via **addition** → gradients flow freely!

---

## Quick Start

### Training
```bash
# Train on text file
python train_minimal.py --data input.txt --epochs 50 --hidden-size 128

# Custom hyperparameters
python train_minimal.py --data input.txt --hidden-size 256 --seq-length 100 --lr 0.001
```

### Generating
```bash
# Generate text from trained model
python train_minimal.py --generate --checkpoint lstm_model.pkl --length 500 --temperature 0.8
```

### In Python
```python
from implementation import LSTM

# Create LSTM
lstm = LSTM(input_size=vocab_size, hidden_size=128, output_size=vocab_size)

# Forward pass
loss = lstm.forward(inputs, targets, h_prev, C_prev)

# Backward pass
dh_next, dC_next = lstm.backward()

# Update weights
lstm.update_weights(learning_rate=0.001)

# Sample text
text = lstm.sample(idx_to_char, seed_char_idx, length=200, temperature=0.8)
```

---

## Hyperparameter Guide

| Parameter      | Typical Range | Description                           | Too Low                | Too High               |
|----------------|---------------|---------------------------------------|------------------------|------------------------|
| `hidden_size`  | 64-512        | Number of LSTM units                  | Can't capture patterns | Overfits, slow         |
| `seq_length`   | 25-100        | Length of training sequences          | No context             | Slow, memory issues    |
| `learning_rate`| 0.0001-0.01   | Step size for weight updates          | Slow learning          | Unstable, diverges     |
| `epochs`       | 10-100        | Full passes through dataset           | Underfits              | Overfits, time waste   |
| `temperature`  | 0.3-1.5       | Sampling randomness (generation only) | Too repetitive         | Too random/nonsensical |

### Good Starting Point
```python
hidden_size = 128
seq_length = 50
learning_rate = 0.001
epochs = 50
temperature = 0.8  # for sampling
```

---

## Common Issues & Fixes

### 1. Loss Explodes (NaN)
**Symptom**: Loss becomes `nan` after few iterations

**Causes**:
- Learning rate too high
- Gradient explosion

**Fixes**:
```python
# Reduce learning rate
learning_rate = 0.0001  # instead of 0.01

# Add gradient clipping (already in implementation.py)
np.clip(grad, -5, 5)
```

### 2. Loss Doesn't Decrease
**Symptom**: Loss stays high, no improvement

**Causes**:
- Learning rate too low
- Hidden size too small
- Forget gate bias not initialized to 1

**Fixes**:
```python
# Increase learning rate
learning_rate = 0.001

# Larger hidden size
hidden_size = 256

# Check forget gate bias (should be 1.0 in __init__)
self.bf = np.ones(hidden_size)  # Remember by default
```

### 3. Generated Text is Repetitive
**Symptom**: Model generates same characters over and over

**Causes**:
- Temperature too low
- Model stuck in local minimum
- Overtrained on small dataset

**Fixes**:
```python
# Increase temperature
temperature = 1.0  # instead of 0.5

# Add diversity
sample = lstm.sample(idx_to_char, seed, length=200, temperature=1.2)

# Train longer or on more data
```

### 4. Training is Slow
**Symptom**: Each iteration takes too long

**Causes**:
- Hidden size too large
- Sequence length too long
- Large vocabulary

**Fixes**:
```python
# Reduce hidden size
hidden_size = 64

# Shorter sequences
seq_length = 25

# Process data in batches (not in current implementation, but possible)
```

---

## Debugging Checklist

When things go wrong, check:

- [ ] **Data loaded correctly?** Print `len(data)`, `vocab_size`
- [ ] **Shapes match?** `input_size == vocab_size`, `output_size == vocab_size`
- [ ] **Forget bias = 1?** Check `lstm.bf` (should be all 1s initially)
- [ ] **Learning rate reasonable?** Try 0.001 first
- [ ] **Gradients clipped?** Should be in `[-5, 5]` range
- [ ] **Loss decreasing?** Plot loss curve, should trend downward
- [ ] **Hidden/cell states reset?** Initialize `h_prev`, `C_prev` to zeros
- [ ] **Temperature for sampling?** Try 0.7-1.0 range

---

## Visualization

```python
from visualization import (
    plot_gate_activations,
    plot_cell_state_evolution,
    plot_gradient_flow_comparison,
    analyze_gate_patterns
)

# Plot what gates are doing
plot_gate_activations(gates_dict, sequence_text)

# See cell state evolution
plot_cell_state_evolution(cell_states, sequence_text)

# Compare gradient flow
plot_gradient_flow_comparison(lstm_grads, rnn_grads)

# Analyze gate patterns
analyze_gate_patterns(gates_dict, sequence_text)
```

---

## When to Use LSTM vs Alternatives

| Model       | Best For                              | Strengths                        | Weaknesses                  |
|-------------|---------------------------------------|----------------------------------|-----------------------------|
| **LSTM**    | Medium sequences (10-100 steps)       | Solves vanishing gradients       | Slower than GRU             |
|             | Interpretable gates needed            | Well-studied, stable             | Can't handle very long seqs |
| **GRU**     | Similar to LSTM                       | Faster (3 gates vs 4)            | Slightly less powerful      |
|             | When speed matters                    | Simpler architecture             |                             |
| **Vanilla RNN** | Very short sequences (<10 steps)  | Simple, fast                     | Vanishing gradients         |
|             | Real-time processing                  | Low memory                       | Can't learn long deps       |
| **Transformer** | Long sequences (100+ steps)       | Parallel processing              | Needs lots of data          |
|             | State-of-the-art performance          | Attention mechanism              | Memory intensive            |

**Rule of thumb**:
- Short sequences (<10): Vanilla RNN
- Medium sequences (10-100): LSTM or GRU
- Long sequences (100+): Transformer
- Very long (1000+): Transformer with sparse attention

---

## Tips & Tricks

### 1. Initialization
```python
# Forget gate bias should be 1 (default to remembering)
self.bf = np.ones(hidden_size)

# Other biases can be 0
self.bi = np.zeros(hidden_size)
self.bc = np.zeros(hidden_size)
self.bo = np.zeros(hidden_size)
```

### 2. Training
- Start with short sequences, gradually increase
- Use gradient clipping (prevents explosion)
- Monitor both loss AND sample quality
- Save checkpoints regularly

### 3. Sampling
- Temperature = 0.5-0.8 for coherent text
- Temperature = 1.0-1.5 for creative text
- Try different seed characters
- Generate longer sequences (500+ chars) to see patterns

### 4. Debugging
- Print gate activations (should be in [0, 1])
- Check cell state magnitude (shouldn't explode)
- Visualize loss curve (should decrease)
- Sample every N iterations to monitor progress

---

## Key Equations Summary

```
Concatenate input:  concat = [h_{t-1}, x_t]

Forget gate:        f_t = σ(W_f · concat + b_f)
Input gate:         i_t = σ(W_i · concat + b_i)
Cell candidate:     C̃_t = tanh(W_C · concat + b_C)
Output gate:        o_t = σ(W_o · concat + b_o)

Update cell:        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Update hidden:      h_t = o_t ⊙ tanh(C_t)

Output:             y_t = softmax(W_y · h_t + b_y)
Loss:               L = -log(y_t[target])
```

**Remember**: 
- σ = sigmoid (0 to 1, for gates)
- tanh = hyperbolic tangent (-1 to 1, for values)
- ⊙ = element-wise multiplication

---

## Resources

- **Paper**: "Understanding LSTM Networks" by Colah
- **Original**: Hochreiter & Schmidhuber (1997)
- **Code**: `implementation.py` (our NumPy version)
- **Exercises**: See `exercises/` folder
- **Visualizations**: `visualization.py`

---

## Quick Comparison: LSTM vs RNN

| Aspect              | Vanilla RNN                | LSTM                           |
|---------------------|----------------------------|--------------------------------|
| **Memory**          | Hidden state only          | Hidden + Cell state            |
| **Gates**           | None                       | 4 (forget, input, cell, output)|
| **Gradient Flow**   | Multiplicative (vanishes)  | Additive (preserved)           |
| **Long-term Deps**  | Poor (<10 steps)           | Good (100+ steps)              |
| **Complexity**      | Simple                     | More complex                   |
| **Training Speed**  | Fast                       | Slower                         |
| **Parameters**      | Few                        | 4x more                        |

---

That's it. For exercises, see the `exercises/` folder.
