# Day 2: Understanding LSTM Networks

> *"Understanding LSTM Networks"* - Christopher Olah (2015)

**Original Post:** http://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Time:** 3-5 hours
**Prerequisites:** Day 1 (vanilla RNN), basic calculus
**Code:** Pure NumPy (no PyTorch needed)

---

## What This Post Is Actually About

Colah's blog post is a visual architecture explainer. He walks through how LSTMs work step by step with diagrams that became the standard reference. If you've seen an LSTM diagram anywhere on the internet, it probably traces back to this post.

The post doesn't cover training, benchmarks, or practical tips — it's purely about understanding the architecture and why it solves the vanishing gradient problem. Those diagrams made LSTMs accessible to everyone.

This matters because LSTMs were the dominant sequence model from ~2015-2017 and understanding their gates is prerequisite for understanding attention mechanisms (Day 13+).

---

## The Core Idea

**Problem:** Vanilla RNNs can't learn long-range dependencies because gradients vanish during backpropagation through time.

**Solution:** Add a **cell state** — a separate path where information flows via addition, not multiplication. Three gates control what gets written, kept, and exposed.

Colah's running example throughout the post: a language model tracking subject gender. When processing "I grew up in France... I speak fluent ___", the cell state preserves "France" across the intervening words so the model can predict "French."

For the full gate-by-gate walkthrough with Colah's own examples, quoted metaphors, and variant discussion, see [paper_notes.md](paper_notes.md).

---

## What Colah Actually Covers

Brief highlights — the detailed treatment is in paper_notes.md:

### The Long-Term Dependency Problem
Two examples: "the clouds are in the ___" (easy, short range) vs. "I grew up in France... I speak fluent ___" (hard, long range). References Hochreiter (1991) and Bengio et al. (1994) on why this is fundamentally difficult.

### The Cell State as "Conveyor Belt"
His central metaphor. The cell state runs parallel to the hidden state with only "minor linear interactions." Information rides along it unchanged unless explicitly modified by gates.

### Three Gates
- **Forget gate**: what to throw away from cell state (sigmoid, 0-1)
- **Input gate + cell candidate**: what new information to write (sigmoid + tanh)
- **Output gate**: what to expose from cell state (sigmoid)

Each explained with the gender-tracking example.

### Variants
Peephole connections (Gers & Schmidhuber 2000), coupled forget/input gates, and GRU (Cho et al. 2014). References Greff et al. (2015) finding variants perform "about the same" and Jozefowicz et al. (2015) testing 10,000+ architectures.

### Conclusion
Attention is the "next step" — written in 2015, two years before Transformers proved him right.

---

## The Architecture

### The LSTM Cell

```
         ┌─────────────────────────────────┐
         │    LSTM Cell at time t          │
         │                                 │
    x_t ──┬──► Forget Gate (f_t)          │
         ││                                │
  h_{t-1}─┼──► Input Gate (i_t)           │
         ││    Cell Candidate (~C_t)       │
         ││                                │
  C_{t-1}─┼──► Cell State Update          │──► C_t
         ││    C_t = f_t ⊙ C_{t-1}        │
         ││         + i_t ⊙ ~C_t           │
         ││                                │
         │└──► Output Gate (o_t)           │
         │     h_t = o_t ⊙ tanh(C_t)      │──► h_t
         └─────────────────────────────────┘
```

### Step-by-Step: What Happens in One Time Step

**Inputs:**
- `x_t` = current input (one-hot encoded character)
- `h_{t-1}` = previous hidden state
- `C_{t-1}` = previous cell state

**Step 1: Forget Gate** — What to forget?
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Outputs values between 0 and 1. A 0 means "completely forget this dimension," 1 means "completely keep."

**Step 2: Input Gate + Cell Candidate** — What new info to store?
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

`i_t` controls how much to let in (0 to 1). `C̃_t` contains candidate values (-1 to 1).

**Step 3: Cell State Update** — Update long-term memory
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

This is the key equation. Forget old info (multiply by f_t), add new info (multiply candidate by i_t). The additive connection from `C_{t-1}` to `C_t` is why gradients don't vanish.

**Step 4: Output Gate** — What to output?
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

Decides which parts of the cell state to expose as output.

### Weight Matrices

For vocabulary size V, hidden size H:

```python
# 4 sets of weights, one per gate (each takes concatenated [h, x])
W_f = np.random.randn(H, V + H) * 0.01  # Forget gate
W_i = np.random.randn(H, V + H) * 0.01  # Input gate
W_C = np.random.randn(H, V + H) * 0.01  # Cell candidate
W_o = np.random.randn(H, V + H) * 0.01  # Output gate

# Biases
b_f = np.ones((H, 1))   # Initialize to 1! (remember by default)
b_i = np.zeros((H, 1))
b_C = np.zeros((H, 1))
b_o = np.zeros((H, 1))
```

That's 4x the parameters of a vanilla RNN. Each weight matrix is (H, V+H) because the input is the concatenation of h_{t-1} and x_t.

The `b_f = 1` initialization is important — it makes the forget gate default to "keep everything" until the model learns otherwise. From Jozefowicz et al. (2015).

### The Vanishing Gradient Solution

**Why Vanilla RNNs fail:**

In BPTT, gradients flow backward:
$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Each term involves the weight matrix $W_{hh}$. If eigenvalues of $W_{hh}$ are < 1, gradients shrink exponentially. After 10 steps with factor 0.9: $0.9^{10} = 0.35$. After 50 steps: $0.9^{50} = 0.005$.

**Why LSTMs work:**

The cell state update gives a direct path:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since $f_t$ is a learned scalar (not a fixed matrix), the LSTM can preserve gradients by learning $f_t \approx 1$ when needed. No matrix multiplication bottleneck.

---

## Implementation Guide

### Forward Pass (One Time Step)

```python
def lstm_step_forward(x, h_prev, C_prev, Wf, Wi, WC, Wo, bf, bi, bC, bo):
    """One LSTM time step."""
    
    # Concatenate input and previous hidden state
    combined = np.vstack([h_prev, x])  # Shape: (H+V, 1)
    
    # 1. Forget gate
    f = sigmoid(Wf @ combined + bf)  # Shape: (H, 1)
    
    # 2. Input gate + candidate
    i = sigmoid(Wi @ combined + bi)
    C_candidate = np.tanh(WC @ combined + bC)
    
    # 3. Cell state update
    C = f * C_prev + i * C_candidate
    
    # 4. Output gate + hidden state
    o = sigmoid(Wo @ combined + bo)
    h = o * np.tanh(C)
    
    # Cache for backward pass
    cache = (x, h_prev, C_prev, f, i, C_candidate, o, C, combined)
    
    return h, C, cache
```

### Backward Pass (One Time Step)

```python
def lstm_step_backward(dh_next, dC_next, cache):
    """Backward pass through one LSTM step."""
    
    x, h_prev, C_prev, f, i, C_cand, o, C, combined = cache
    
    # Output gate gradients
    do = dh_next * np.tanh(C)
    do_raw = do * o * (1 - o)  # sigmoid derivative
    
    # Cell state gradient (from output + from future)
    dC = dh_next * o * (1 - np.tanh(C)**2) + dC_next
    
    # Forget gate gradients
    df = dC * C_prev
    df_raw = df * f * (1 - f)
    
    # Input gate gradients
    di = dC * C_cand
    di_raw = di * i * (1 - i)
    
    # Cell candidate gradients
    dC_cand = dC * i
    dC_cand_raw = dC_cand * (1 - C_cand**2)
    
    # Weight gradients
    dW_f = df_raw @ combined.T
    dW_i = di_raw @ combined.T
    dW_C = dC_cand_raw @ combined.T
    dW_o = do_raw @ combined.T
    
    # Bias gradients
    db_f, db_i, db_C, db_o = df_raw, di_raw, dC_cand_raw, do_raw
    
    # Gradients to pass backward
    dcombined = (Wf.T @ df_raw + Wi.T @ di_raw + 
                 WC.T @ dC_cand_raw + Wo.T @ do_raw)
    
    dh_prev = dcombined[:H]
    dx = dcombined[H:]
    dC_prev = dC * f  # Cell state gradient flows through forget gate
    
    return dx, dh_prev, dC_prev, dW_f, dW_i, dW_C, dW_o, db_f, db_i, db_C, db_o
```

The key line: `dC_prev = dC * f` — the gradient through the cell state is just multiplied by the forget gate, not by a weight matrix. This is why LSTMs preserve gradients.

---

## Implementation Notes

The implementation in `implementation.py` is a from-scratch LSTM in pure NumPy, analogous to Day 1's vanilla RNN.

Key differences from Day 1:
- **4 gate weight matrices** instead of 1 (forget, input, cell candidate, output)
- **Cell state** carried alongside hidden state
- **Forget gate bias = 1** (not 0) for stable training
- **4x more parameters** — LSTMs are slower but handle longer sequences

Things that will bite you:
- **Forgetting the cell state** — you need to pass both `h` and `C` between steps. Easy to forget `C` and wonder why it doesn't learn.
- **Gate ordering** — the equations look similar. Common bug: using the wrong weight matrix for the wrong gate.
- **Gradient clipping still needed** — LSTMs fix vanishing gradients but can still explode. Clip to [-5, 5].
- **Numerical stability** — same softmax trick as Day 1 (subtract max before exp).

---

## Training Tips

### 1. Initialization Matters

```python
# Forget gate bias = 1 (Jozefowicz et al., 2015)
b_f = np.ones((hidden_size, 1))

# Xavier initialization for weights
W_f = np.random.randn(H, V+H) * np.sqrt(2.0 / (V+H))
```

Starting with forget gate near 1 means "remember by default" until the model learns otherwise.

### 2. Gradient Clipping Still Needed

```python
for grad in [dW_f, dW_i, dW_C, dW_o]:
    np.clip(grad, -5, 5, out=grad)
```

LSTMs reduce vanishing but can still explode.

### 3. Learning Rate

```python
learning_rate = 0.001  # Start lower than vanilla RNN
```

LSTMs have 4x the parameters, so they're more sensitive to learning rate.

### 4. Sequence Length

```python
seq_length = 50  # LSTMs can handle longer sequences
```

Unlike vanilla RNNs (seq_length ~20), LSTMs work well with 50-100 steps.

---

## Visualizations

### 1. Gate Activation Patterns

Plot forget/input/output gate values over a sequence to see what the LSTM actually learns:

```python
gates = {'forget': [], 'input': [], 'output': []}

for t in range(seq_len):
    h, C, cache = lstm_step_forward(...)
    f, i, o = cache[3], cache[4], cache[6]
    gates['forget'].append(f)
    gates['input'].append(i)
    gates['output'].append(o)

plt.imshow(np.array(gates['forget']).T, cmap='RdYlGn', aspect='auto')
plt.title('Forget Gate (0=forget, 1=keep)')
```

### 2. Cell State Evolution

```python
cell_states = np.array(all_cell_states).T  # (hidden_size, seq_len)
plt.imshow(cell_states, cmap='RdBu', aspect='auto')
plt.colorbar(label='Cell state value')
plt.title('LSTM Cell State Evolution')
```

### 3. Gradient Flow Comparison

```python
plt.plot(lstm_grads, label='LSTM', linewidth=2)
plt.plot(rnn_grads, label='Vanilla RNN', linewidth=2, alpha=0.7)
plt.yscale('log')
plt.xlabel('Time step (backward)')
plt.ylabel('Gradient norm')
plt.title('Gradient Flow: LSTM vs RNN')
```

---

## What to Build

### Quick Start

```bash
python train_minimal.py --data data/input.txt --epochs 50 --hidden-size 128
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Build LSTM from scratch | Understand the 4-gate forward pass and BPTT |
| 2 | Gate analysis | Visualize what forget/input/output gates learn |
| 3 | Ablation study | Remove gates one at a time — which matters most? |
| 4 | Long-range dependencies | Synthetic task: LSTM succeeds where vanilla RNN fails |
| 5 | GRU comparison | Implement GRU (2 gates), compare with LSTM |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Going Further

### LSTM Variants

1. **Peephole Connections** (Gers & Schmidhuber, 2000)
   - Gates can look at the cell state directly
   - Slightly better on some tasks, not universally adopted

2. **GRU** (Cho et al., 2014)
   - Merges forget and input into a single "update gate"
   - Merges cell state and hidden state
   - 2 gates instead of 3 — fewer parameters, often comparable performance
   - Exercise 5 implements this

3. **Bidirectional LSTM**
   - Process sequence forward AND backward
   - Used in BERT, ELMo

### When to Use LSTMs vs Alternatives

| Model | Best For | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **Vanilla RNN** | Very short sequences (<10 steps) | Simple, fast | Vanishing gradients |
| **LSTM** | Medium sequences (10-100 steps) | Solves vanishing gradients, interpretable gates | Slower than GRU, sequential |
| **GRU** | Similar to LSTM | Faster (fewer parameters) | Slightly less expressive |
| **Transformer** | Long sequences (100+ steps) | Parallel training, attention | Needs more data, memory intensive |

Rule of thumb: short sequences → RNN, medium → LSTM/GRU, long → Transformer.

---

## Key Takeaways

1. **The cell state is the key innovation.** It provides an additive path for gradients to flow backward through time without vanishing.

2. **Gates are learned, not hand-crafted.** The network discovers when to remember and when to forget through training.

3. **Variants don't matter much.** Greff et al. (2015) compared popular LSTM variants and found them roughly equivalent. The core design (additive cell state + learned gates) is what matters.

4. **This is the bridge to attention.** Colah correctly predicted attention was the "next step" — it was originally added on top of LSTMs before replacing them entirely (Transformers, Day 13+).

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Complete LSTM in NumPy, heavily commented |
| `train_minimal.py` | Training script with CLI args |
| `visualization.py` | Gate activation heatmaps, cell state plots, gradient flow comparison |
| `notebook.ipynb` | Interactive walkthrough — build, train, visualize |
| `exercises/` | 5 exercises with solutions |
| `paper_notes.md` | Detailed notes on Colah's post with ELI5 |
| `CHEATSHEET.md` | Quick reference for hyperparameters and debugging |

---

## Further Reading

- [Colah's Blog Post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) — read this first, the diagrams are essential
- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) — Hochreiter & Schmidhuber (1997), the original
- [Learning to Forget](https://ieeexplore.ieee.org/document/818041) — Gers et al. (2000), added the forget gate
- [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069.pdf) — Greff et al. (2015), comparison of variants
- [Jozefowicz et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) — tested 10,000+ RNN architectures
- [PyTorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — production implementation

---

**Next:** [Day 3 — RNN Regularization](../03_RNN_Regularization/)
