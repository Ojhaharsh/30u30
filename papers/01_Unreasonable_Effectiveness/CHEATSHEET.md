# Day 1 Cheat Sheet: Character-Level RNN

Quick reference for training and using your RNN.

---

## Quick Start

```bash
# Train on sample data
python train_minimal.py --data data/tiny_shakespeare.txt --epochs 200

# Train with custom settings
python train_minimal.py --data your_data.txt --epochs 500 --hidden-size 128 --seq-length 50 --lr 0.01

# Generate text from trained model
python train_minimal.py --generate --checkpoint model.pkl --length 500 --temperature 0.8
```

---

## Key Hyperparameters

| Parameter | Typical Range | What It Does | Tips |
|-----------|--------------|--------------|------|
| `hidden_size` | 50-200 | Size of RNN memory | Bigger = more capacity, slower |
| `seq_length` | 10-100 | Characters per batch | Longer = more context, slower |
| `learning_rate` | 0.01-0.5 | Update step size | Start 0.1, reduce if unstable |
| `temperature` | 0.2-2.0 | Sampling randomness | 0.8 = balanced, 2.0 = creative |

---

## Common Issues & Fixes

### Loss explodes (NaN)
```python
# Fix: Gradient clipping (already in code)
np.clip(grad, -5, 5, out=grad)
# Or reduce learning rate
```

### Loss plateaus too high
```python
# Fix: Increase hidden_size or train longer
python train_minimal.py --hidden-size 200 --epochs 1000
```

### Generated text is gibberish
```python
# Fix: Train longer or use lower temperature
python train_minimal.py --generate --temperature 0.5
```

### Training is too slow
```python
# Fix: Reduce hidden_size or seq_length
python train_minimal.py --hidden-size 50 --seq-length 25
```

---

## The Math (Copy-Paste Ready)

### Forward Pass
```python
# Hidden state
h_t = np.tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)

# Output
y_t = Why @ h_t + by

# Probabilities
p_t = softmax(y_t)
```

### Backward Pass (BPTT)
```python
# Output gradient
dy = p - y_true

# Hidden gradient
dh = Why.T @ dy + dh_next
dh_raw = (1 - h**2) * dh  # Through tanh

# Clip!
np.clip(grad, -5, 5, out=grad)
```

---

## Visualization Examples

```python
from visualization import *

# Plot training loss
plot_loss_curve(losses, save_path='loss.png')

# Visualize hidden states
plot_hidden_states(hidden_states, "hello world", save_path='states.png')

# Compare temperatures
samples = {0.5: "...", 1.0: "...", 1.5: "..."}
plot_temperature_comparison(samples, save_path='temps.png')
```

---

## Experiment Ideas

### Easy
- Train on different texts (songs, code, tweets)
- Try different temperatures (0.1 to 3.0)
- Vary hidden_size (25, 50, 100, 200)

### Medium  
- Implement learning rate decay
- Add gradient norm monitoring
- Track character-level accuracy

### Advanced
- Multi-layer RNN (stack multiple RNNs)
- Bidirectional RNN (forward + backward)
- Beam search instead of greedy sampling

---

## File Reference

| File | Use It For |
|------|-----------|
| `README.md` | Full tutorial |
| `paper_notes.md` | Quick review |
| `implementation.py` | Reference code |
| `train_minimal.py` | Training |
| `visualization.py` | Plotting |
| `notebook.ipynb` | Interactive learning |
| `exercises/` | Practice problems |

---

## Debugging Checklist

- [ ] Data file exists and is readable?
- [ ] Vocabulary size < 100? (more = harder)
- [ ] Data size > 100KB? (less = won't learn much)
- [ ] Loss starts around `-log(1/vocab_size) * seq_length`?
- [ ] Loss decreases over first 100 iterations?
- [ ] Gradients clipped (no NaN in loss)?
- [ ] Generated samples improve after 500+ iterations?

---

## Pro Tips

1. **Start small**: Test on tiny data first
2. **Monitor loss**: Should decrease smoothly
3. **Sample often**: Generate text every 100 iterations
4. **Save checkpoints**: Don't lose your progress!
5. **Try different data**: Code, lyrics, dialogue - variety helps learning
6. **Visualize**: Plot loss, hidden states, gradients
7. **Temperature matters**: 0.8 is usually best starting point
8. **Be patient**: Good results take 1000+ iterations

---

## Success Criteria

After training, you should see:
- Loss < 1.5 (depends on data)
- Generated words are real
- Basic grammar/structure
- Style matches training data
- No repetitive loops

---

## Next: Day 2

Once you understand RNNs, move to **LSTMs**:
- Solve vanishing gradient problem
- Better long-range memory
- Industry standard (before transformers)

---

**Questions?** Check [README.md](README.md) or ask [GitHub Discussions](../../discussions)

**Share your results!** #30u30
