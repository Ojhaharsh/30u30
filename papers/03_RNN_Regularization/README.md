# Day 3: Recurrent Neural Network Regularization

> Zaremba, Sutskever & Vinyals (2014) — [arXiv:1409.2329](https://arxiv.org/abs/1409.2329)

**Time:** 2-4 hours
**Prerequisites:** Day 1 (RNN basics), Day 2 (LSTM internals)
**Code:** NumPy (builds on Day 2's LSTM)

---

## What This Paper Is Actually About

This paper answers one specific question: **how do you apply dropout to LSTMs without breaking them?**

Standard dropout was a huge success for feedforward networks (Hinton et al., 2012), but people struggled to make it work with RNNs. Bayer et al. (2013) even published results claiming dropout doesn't help recurrent networks. Zaremba, Sutskever, and Vinyals show that's wrong — the problem wasn't dropout itself, but *where* you apply it.

Their solution: apply dropout only to the **non-recurrent** (vertical) connections, never to the **recurrent** (horizontal) connections. This simple rule lets you use dropout to regularize LSTMs while preserving their ability to remember across long sequences.

The result: PTB language modeling perplexity drops from 114.5 (no regularization) to 78.4 (with dropout), demolishing the previous state-of-the-art of 107.5.

---

## The Core Idea

### Why Naive Dropout Fails on RNNs

In a multi-layer LSTM, information flows in two directions:
- **Vertically** (between layers): `h_t^{l-1}` feeds into layer `l` at the same timestep
- **Horizontally** (across time): `h_{t-1}^l` carries memory forward within layer `l`

```
          Timestep 1      Timestep 2      Timestep 3

Layer 2:    h_1^2  ------->  h_2^2  ------->  h_3^2   (horizontal = recurrent)
              ^                ^                ^
              | (vertical)     | (vertical)     |
Layer 1:    h_1^1  ------->  h_2^1  ------->  h_3^1   (horizontal = recurrent)
              ^                ^                ^
              |                |                |
Input:       x_1              x_2              x_3
```

If you apply dropout to the horizontal arrows (recurrent connections), you corrupt the memory at every timestep. Over a 35-step sequence, the signal gets randomly perturbed 35 times — making it nearly impossible for the LSTM to learn long-range dependencies.

### The Fix: Dropout Only on Vertical Connections

The paper's regularized LSTM equations differ from the standard ones by exactly one thing — the dropout operator `D()` wrapping the non-recurrent input:

**Standard:**
```
gates = W * [h_t^{l-1}, h_{t-1}^l] + b
```

**Regularized (this paper):**
```
gates = W * [D(h_t^{l-1}), h_{t-1}^l] + b
```

That's the entire contribution. `D()` is applied to:
- The input embedding at the bottom: `D(x_t)`
- The inter-layer connections: `D(h_t^{l-1})`
- The output to softmax at the top: `D(h_t^L)`

It is NOT applied to:
- The recurrent hidden state: `h_{t-1}^l` passes through clean

### The L+1 Property

A piece of information traveling from input to output encounters dropout exactly **L+1** times (L layers + output), regardless of how many timesteps it persists. This bounded corruption is what makes it work — the noise level doesn't scale with sequence length.

---

## What the Paper Showed

### Penn Treebank Results

| Model | Hidden | Dropout | Test Perplexity |
|-------|--------|---------|-----------------|
| Non-regularized LSTM | 200 | 0% | 114.5 |
| **Medium regularized** | **650** | **50%** | **82.7** |
| **Large regularized** | **1500** | **65%** | **78.4** |
| Previous SOTA (Pascanu 2013) | -- | -- | 107.5 |

Key observations from the paper:
- Without dropout, bigger models overfit and get WORSE
- With dropout, bigger models consistently improve
- The large model beats the previous best by ~30 perplexity points

### Other Tasks

The paper also demonstrates the technique on:
- **Speech recognition** (Icelandic): improved word error rate
- **Machine translation** (WMT'14 EN-FR): BLEU 29.03 vs 25.87 without dropout (4-layer, 1000 hidden, 20% dropout)
- **Image captioning** (MS COCO): improved over non-regularized baseline

---

## The Architecture

### Forward Pass (Single Timestep, Layer l)

```python
# Standard LSTM gates
combined = np.vstack([h_prev_layer, h_prev_time])  # [D(h_t^{l-1}), h_{t-1}^l]

f = sigmoid(Wf @ combined + bf)     # Forget gate
i = sigmoid(Wi @ combined + bi)     # Input gate
g = tanh(Wc @ combined + bc)        # Candidate
o = sigmoid(Wo @ combined + bo)     # Output gate

c_t = f * c_prev + i * g            # Cell state update
h_t = o * tanh(c_t)                 # Hidden state
```

The critical difference from Day 2's LSTM: the input `h_prev_layer` has dropout applied to it BEFORE entering the gate computations. The recurrent input `h_prev_time` does not.

### Dropout Application Points

```python
# Layer 1: dropout on input embedding
x_dropped = dropout(x_t, keep_prob=0.5)
h_1 = lstm_layer_1(x_dropped, h_prev_1, c_prev_1)

# Layer 2: dropout on layer 1 output (non-recurrent connection)
h_1_dropped = dropout(h_1, keep_prob=0.5)
h_2 = lstm_layer_2(h_1_dropped, h_prev_2, c_prev_2)

# Output: dropout on top layer output
h_2_dropped = dropout(h_2, keep_prob=0.5)
y = softmax(Why @ h_2_dropped + by)
```

Notice: dropout is applied 3 times total (L+1 = 2+1 = 3) for a 2-layer LSTM.

### Training Hyperparameters (From the Paper)

**Medium model (recommended starting point):**
```python
num_layers = 2
hidden_size = 650
dropout_keep_prob = 0.5
learning_rate = 1.0       # SGD
lr_decay = 1.2            # Decay after epoch 6
gradient_clip = 5.0
bptt_steps = 35
batch_size = 20
epochs = 39
```

**Large model:**
```python
num_layers = 2
hidden_size = 1500
dropout_keep_prob = 0.35   # More aggressive dropout
learning_rate = 1.0
lr_decay = 1.15            # Gentler decay, starts epoch 14
gradient_clip = 10.0
bptt_steps = 35
batch_size = 20
epochs = 55
```

---

## Implementation Notes

Our implementation in `implementation.py` goes beyond the paper to include additional regularization techniques (layer normalization, weight decay, early stopping) that are commonly used alongside dropout in practice. These are clearly labeled as our additions.

Key decisions:
- **Pure NumPy**: No framework — you see every operation
- **Single-layer LSTM with dropout**: Simpler than the paper's 2-layer setup, but the dropout placement principle is the same
- **Inverted dropout**: We scale by `1/keep_prob` during training so test time needs no modification
- **Layer norm and weight decay**: Not from this paper, but useful pedagogical additions

Things to watch for:
- **Don't apply dropout during evaluation.** The paper is explicit: dropout is training-only. At test time, all neurons are active.
- **Dropout mask is fresh each timestep** (for non-recurrent connections). Gal & Ghahramani (2016) later showed that using the same mask across timesteps works even better (variational dropout), but this paper doesn't do that.
- **The keep_prob values are per-connection.** 50% dropout means each non-recurrent connection independently has a 50% chance of being zeroed at each timestep.

---

## What to Build

### Quick Start

```bash
python train_minimal.py --dropout 0.8 --epochs 15
```

### Exercises (in `exercises/`)

| # | Task | What You'll Learn |
|---|------|-------------------|
| 1 | Implement dropout forward/backward | Core of the paper: how dropout works mechanically |
| 2 | Implement layer normalization | Our addition: stabilizing activations (Ba et al. 2016) |
| 3 | Implement weight decay (L2) | Our addition: penalizing large weights |
| 4 | Implement early stopping | Our addition: stopping before overfitting |
| 5 | Full regularized pipeline | Combine all techniques into one training loop |

Exercises 1 is directly from what the paper covers. Exercises 2-4 are standard regularization techniques that complement dropout. Exercise 5 brings everything together.

Solutions are in `exercises/solutions/`. Try first.

---

## Key Takeaways

1. **Dropout works for LSTMs — but only on non-recurrent connections.** This is the paper's single contribution, and it's an important one. Applying dropout to recurrent connections destroys the LSTM's ability to carry information across time.

2. **The corruption is bounded.** Information encounters dropout exactly L+1 times regardless of sequence length. This property is what makes the approach work for long sequences.

3. **Regularization unlocks scale.** Without dropout, making the LSTM bigger hurts (overfitting). With dropout, bigger is better. This was one of the early demonstrations that regularization and scale work together.

4. **Simple ideas can be powerful.** The entire contribution is one line of change in the LSTM equations. But it enables a 30-point perplexity improvement and generalizes across language modeling, translation, speech, and captioning.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Regularized LSTM in NumPy: dropout + layer norm + weight decay |
| `train_minimal.py` | Training script with CLI args for regularization hyperparameters |
| `visualization.py` | Learning curves, dropout effects, weight distributions |
| `notebook.ipynb` | Interactive walkthrough of regularization techniques |
| `exercises/` | 5 exercises: dropout (from paper), layer norm, weight decay, early stopping, full pipeline |
| `paper_notes.md` | Detailed notes on the actual Zaremba et al. paper |
| `CHEATSHEET.md` | Quick reference for regularization techniques and hyperparameters |

---

## Further Reading

- [Zaremba et al. (2014)](https://arxiv.org/abs/1409.2329) — this paper
- [Gal & Ghahramani (2016)](https://arxiv.org/abs/1512.05287) — variational dropout: same mask across timesteps, theoretically grounded
- [Merity et al. (2018)](https://arxiv.org/abs/1708.02182) — AWD-LSTM: combines many regularization tricks, PTB perplexity ~57
- [Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html) — the original dropout paper
- [Ba et al. (2016)](https://arxiv.org/abs/1607.06450) — layer normalization

---

**Next:** [Day 4 -- Sequence to Sequence Learning](../04_Sequence_to_Sequence/)
