# Day 15 Cheat Sheet: Bahdanau Attention

Quick reference for Neural Machine Translation with Additive Attention.

---

## The Big Idea (30 seconds)

Bahdanau Attention (2014) solves the **bottleneck problem** in sequence-to-sequence (seq2seq) models. Traditional models compress the entire input into a single fixed-length vector, which fails for long sequences. Attention allows the decoder to "look back" at all encoder hidden states at each step, computing a context vector that dynamically focuses on relevant parts of the source sentence.

---

## Quick Start

```bash
# Train on sequence reversal task
python train_minimal.py --task reversal --epochs 20

# Train with custom hidden size
python train_minimal.py --hidden_size 256 --num_layers 2 --lr 0.001

# Visualize attention for a specific sequence
python visualization.py --demo
```

---

## Key Hyperparameters

| Parameter | Typical Range | What It Does | Tips |
|-----------|--------------|--------------|------|
| `hidden_size` | 128-512 | Size of GRU hidden states | Bigger captures longer sequences |
| `attention_dim` | 64-256 | Hidden layer of alignment MLP | Should be similar to hidden_size |
| `learning_rate`| 0.001-0.0001 | Adam optimizer step size | Reduce if loss is unstable |
| `teacher_forcing`| 0.5-1.0 | Ratio of using ground truth | Start at 1.0 for faster convergence |

---

## Common Issues & Fixes

### Loss is NaN or explosions
```python
# Fix: Implement gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Also: Ensure padding mask uses -1e9/inf BEFORE softmax
```

### Attention weights are uniform
```python
# Fix: Train longer. Attention is learned and takes time to focus.
# Or: Increase hidden_size if the task is too complex.
```

### Model ignores end-of-sequence
```python
# Fix: Ensure <eos> token is included in target during training
# And: Stop generation in translate() when <eos> is sampled
```

---

## The Math (Paper Grounding)

### Alignment Score (Additive)
$$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

### Attention Weights
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^N \exp(e_{ik})}$$

### Context Vector
$$c_i = \sum_{j=1}^N \alpha_{ij} h_j$$

---

## Visualization Examples

```python
from visualization import plot_attention_heatmap

# Visualize the alignment matrix
plot_attention_heatmap(weights, src_tokens, trg_tokens, save_path='attn.png')

# Check for diagonal pattern in reversal task
```

---

## Experiment Ideas

- Compare performance on length 5 vs length 20 sequences.
- Vary the teacher forcing ratio.
- Replace Bahdanau (additive) with Luong (multiplicative) attention.
- Add learning rate scheduling (ReduceLROnPlateau).

---

## File Reference

| File | Use It For |
|------|-----------|
| `README.md` | Full tutorial and curriculum context |
| `paper_notes.md` | Theoretical deep dive and paper review |
| `implementation.py` | Core model classes and logic |
| `train_minimal.py` | Training loops and orchestration |
| `visualization.py` | Plotting heatmaps and metrics |
| `notebook.ipynb` | Interactive step-by-step learning |
| `exercises/` | Hands-on implementation practice |

---

## Debugging Checklist

- [ ] Is the padding mask applied correctly?
- [ ] Are encoder bidirectional outputs concatenated (not summed)?
- [ ] Is teacher forcing handled separately for training vs evaluation?
- [ ] Does the attention weight sum to 1.0 for every decoder step?
- [ ] Are you using `ignore_index` in the loss function for padding?

---

## Pro Tips

1. **Mask Early**: Setting scores to $-\infty$ before softmax is critical for stability.
2. **Context Injection**: Concatenate the context vector with the input *and* the GRU output for best results.
3. **BiRNN**: Always use a bidirectional encoder; it provides the "future" context necessary for valid alignments.
4. **Log Attention**: Always visualize the attention weights; they are your best debugging tool.

---

## Success Criteria

After training, you should see:
- A clear "reversed diagonal" pattern in the reversal task.
- Accuracy > 95% on sequences up to length 15.
- Attention entropy decreasing over training epochs.

---

## Next: Day 16

Once you understand basic attention, move to **Pointer Networks**:
- Handle variable-sized output spaces.
- "Point" to input elements instead of a fixed vocabulary.
- Solving the Traveling Salesman Problem (TSP).

---

**Questions?** Check [README.md](README.md)
