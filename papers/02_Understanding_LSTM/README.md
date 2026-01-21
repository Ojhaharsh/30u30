# Day 2: Understanding LSTM Networks

> *"Understanding LSTM Networks"* - Christopher Olah (2015)

**📖 Original Post:** http://colah.github.io/posts/2015-08-Understanding-LSTMs/

**⏱️ Time to Complete:** 3-5 hours

**🎯 What You'll Learn:**
- Why vanilla RNNs fail on long sequences
- How memory cells work
- The three gates: forget, input, output
- Why LSTMs conquered sequence modeling (before transformers)

---

## 🧠 The Big Idea

**In one sentence:** LSTMs add a "memory cell" with gates that control what to remember, what to forget, and what to output—solving the vanishing gradient problem that plagued vanilla RNNs.

### The Problem with Vanilla RNNs

Remember Day 1? Character-level RNNs work great for short-term patterns. But they have a critical flaw:

**Vanishing Gradients** = The further back in time, the weaker the learning signal becomes.

Imagine trying to learn this sentence:
```
"The cat, which was very fluffy and had been sleeping all day, was hungry."
```

A vanilla RNN struggles to connect "cat" with "was hungry" because there are 12 words in between. The gradient has to flow backward through all those steps, getting weaker each time (multiplied by values < 1).

### The LSTM Solution

LSTMs add a **cell state** (C) that runs parallel to the hidden state (h). Think of it as a **conveyor belt** that information can ride on unchanged, unless the LSTM explicitly decides to modify it.

```
Regular RNN: h → h → h → h → h (gradient weakens)
     LSTM:   C →→→→→→→→→→→→→→ C (gradient flows easily)
             h → h → h → h → h
```

The cell state can preserve information across many time steps without degradation!

---

## 🤔 Why "Understanding" LSTMs?

This isn't just another architecture—it's a **fundamental insight** about machine learning:

**You can design neural networks that learn WHEN to remember and WHEN to forget.**

Before LSTMs:
- ❌ RNNs couldn't learn long-range dependencies
- ❌ Gradients vanished after ~10 steps
- ❌ Simple tasks like "remember the first word" were impossible

After LSTMs:
- ✅ Machine translation became viable
- ✅ Speech recognition improved dramatically
- ✅ Text generation got coherent over long passages
- ✅ Opened the door to seq2seq models

Christopher Olah's blog post made LSTMs accessible to everyone. His visualizations became the standard way people understand this architecture.

---

## 🌍 Real-World Analogy

### The Todo List Analogy

Imagine you're managing a todo list throughout your day:

**Forget Gate** = Crossing items off
- Morning: "buy milk" ✓ (done, forget it)
- Afternoon: Still remember "pick up kids at 3pm"

**Input Gate** = Adding new items
- Boss calls: "New task: send report by 5pm" (add to list)
- Friend texts: "Want to grab coffee?" (maybe ignore)

**Output Gate** = Deciding what's relevant NOW
- At 2:55pm: "Pick up kids" becomes the ONLY thing you think about
- Everything else is still on the list, but not active

That's exactly how LSTM gates work:
- **Forget gate**: What old memories can we discard?
- **Input gate**: What new information should we store?
- **Output gate**: What should we focus on right now?

### The Water Pipeline Analogy

Think of the cell state as a **water pipeline**:

```
Input → [Valve 1: Forget] → Pipeline → [Valve 2: Input] → → → [Valve 3: Output] → Output
                                ↓
                          Memory flows
```

- **Valve 1 (Forget)**: Opens to drain old water (forget old info)
- **Valve 2 (Input)**: Opens to add fresh water (add new info)
- **Valve 3 (Output)**: Controls how much water goes to output (what to use now)

The pipeline itself can carry water for miles unchanged—that's the cell state!

---

## 📊 The Architecture

### The Four Components

```
         ┌─────────────────────────────────┐
         │    LSTM Cell at time t          │
         │                                 │
    x_t ──┬──► Forget Gate (f_t)          │
         ││                                │
    h_{t-1}┼──► Input Gate (i_t)          │
         ││    Cell Candidate (~C_t)      │
         ││                                │
    C_{t-1}┼──► Cell State Update         │──► C_t
         ││    C_t = f_t ⊙ C_{t-1}       │
         ││         + i_t ⊙ ~C_t          │
         ││                                │
         │└──► Output Gate (o_t)          │
         │     h_t = o_t ⊙ tanh(C_t)      │──► h_t
         └─────────────────────────────────┘
```

### Step-by-Step: What Happens in One Time Step

**Input:** 
- `x_t` = current input (e.g., word embedding)
- `h_{t-1}` = previous hidden state
- `C_{t-1}` = previous cell state

**Step 1: Forget Gate** (What to forget?)
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- Outputs values between 0 and 1 (sigmoid)
- 0 = "completely forget"
- 1 = "completely keep"
- Example: Forget the subject when a new sentence starts

**Step 2: Input Gate** (What new info to store?)
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- `i_t` = how much to let in (0 to 1)
- `~C_t` = candidate values to add (-1 to 1)
- Example: Store new subject "The dog"

**Step 3: Cell State Update** (Update long-term memory)
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

- `⊙` = element-wise multiplication
- Forget old info, add new info
- This is the **key**: direct connection from `C_{t-1}` to `C_t`!

**Step 4: Output Gate** (What to output?)
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

- Decides which parts of cell state to expose
- Example: Output "dog" when predicting verb for "The dog..."

---

## 💡 The Vanishing Gradient Solution

### Why Vanilla RNNs Fail

In backpropagation through time (BPTT), gradients flow like this:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Each term $\frac{\partial h_t}{\partial h_{t-1}}$ involves the weight matrix $W_{hh}$.

If eigenvalues of $W_{hh}$ are:
- **< 1**: Gradients shrink exponentially → **vanishing**
- **> 1**: Gradients explode exponentially → **explosion**

After just 10 steps with factor 0.9: $0.9^{10} = 0.35$ (gradient is 1/3 original)

### Why LSTMs Work

The cell state update has a **direct path**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

When we backpropagate:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since $f_t$ is learned (not a fixed matrix), the LSTM can:
- **Preserve gradients** by learning $f_t \approx 1$ when needed
- **Block gradients** by learning $f_t \approx 0$ when not needed

This is like having a **highway** for gradients that bypasses the multiplication bottleneck!

---

## 🎨 Visualizing LSTM Behavior

### Example: Learning Name Gender

Input: "Sarah went to the store. She bought..."

**Time step 1:** "Sarah"
- **Input gate**: OPEN (store "Sarah" + "female")
- **Forget gate**: OPEN (forget previous context)
- **Cell state**: [female, singular, subject=Sarah]

**Time steps 2-6:** "went to the store"
- **Input gate**: mostly CLOSED (nothing important to store)
- **Forget gate**: mostly OPEN (keep "Sarah" info)
- **Cell state**: [female, singular, subject=Sarah] (preserved!)

**Time step 7:** "She"
- **Output gate**: OPEN (output "female" to predict "She")
- Model correctly uses "She" instead of "He"!

The cell state remembered "Sarah = female" for 6 steps!

---

## 🔧 Implementation Guide

### Weight Matrices

For vocabulary size V, hidden size H:

```python
# Input to gates (4 sets of weights, one per gate)
W_f = np.random.randn(H, V + H) * 0.01  # Forget gate
W_i = np.random.randn(H, V + H) * 0.01  # Input gate
W_C = np.random.randn(H, V + H) * 0.01  # Cell candidate
W_o = np.random.randn(H, V + H) * 0.01  # Output gate

# Biases
b_f = np.ones((H, 1))   # Bias toward remembering (forget gate = 1)
b_i = np.zeros((H, 1))
b_C = np.zeros((H, 1))
b_o = np.zeros((H, 1))
```

**Note:** `b_f` initialized to 1 (forget gate bias) encourages remembering by default!

### Forward Pass (One Time Step)

```python
def lstm_step_forward(x, h_prev, C_prev, Wf, Wi, WC, Wo, bf, bi, bC, bo):
    """One LSTM time step"""
    
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

### Backward Pass (BPTT)

The beauty of LSTMs: gradients flow through the cell state with minimal degradation!

```python
def lstm_step_backward(dh_next, dC_next, cache):
    """Backward pass through one LSTM step"""
    
    x, h_prev, C_prev, f, i, C_cand, o, C, combined = cache
    
    # Gradient flowing into this step
    dh = dh_next
    
    # Output gate gradients
    do = dh * np.tanh(C)
    do_raw = do * o * (1 - o)  # Sigmoid derivative
    
    # Cell state gradient (from output + from future)
    dC = dh * o * (1 - np.tanh(C)**2) + dC_next
    
    # Forget gate gradients
    df = dC * C_prev
    df_raw = df * f * (1 - f)
    
    # Input gate gradients
    di = dC * C_cand
    di_raw = di * i * (1 - i)
    
    # Cell candidate gradients
    dC_cand = dC * i
    dC_cand_raw = dC_cand * (1 - C_cand**2)
    
    # Weight gradients (all gates)
    dW_f = df_raw @ combined.T
    dW_i = di_raw @ combined.T
    dW_C = dC_cand_raw @ combined.T
    dW_o = do_raw @ combined.T
    
    # Bias gradients
    db_f = df_raw
    db_i = di_raw
    db_C = dC_cand_raw
    db_o = do_raw
    
    # Gradients to pass backward
    dcombined = (Wf.T @ df_raw + Wi.T @ di_raw + 
                 WC.T @ dC_cand_raw + Wo.T @ do_raw)
    
    dh_prev = dcombined[:H]
    dx = dcombined[H:]
    dC_prev = dC * f  # Cell state gradient to previous step
    
    return dx, dh_prev, dC_prev, dW_f, dW_i, dW_C, dW_o, db_f, db_i, db_C, db_o
```

**Key insight:** `dC_prev = dC * f` means the gradient flows through with minimal attenuation!

---

## 🎯 Training Tips

### 1. **Initialization Matters**

```python
# Forget gate bias = 1 (Jozefowicz et al., 2015)
b_f = np.ones((hidden_size, 1))

# Xavier initialization for weights
W_f = np.random.randn(H, V+H) * np.sqrt(2.0 / (V+H))
```

**Why?** Starting with forget gate ≈ 1 means "remember by default" until the model learns otherwise.

### 2. **Gradient Clipping Still Needed**

```python
# Clip gradients to prevent explosion
for grad in [dW_f, dW_i, dW_C, dW_o]:
    np.clip(grad, -5, 5, out=grad)
```

LSTMs reduce vanishing, but can still explode!

### 3. **Learning Rate**

```python
learning_rate = 0.001  # Start lower than vanilla RNN
```

LSTMs have 4× the parameters, so they're more sensitive.

### 4. **Sequence Length**

```python
seq_length = 50  # LSTMs can handle longer sequences!
```

Unlike vanilla RNNs (seq_length ≈ 20), LSTMs work well with 50-100 steps.

---

## 📈 Visualizations

### 1. Gate Activation Patterns

Plot forget/input/output gate values over a sequence to see what the LSTM learns:

```python
# After training, run forward pass and collect gates
gates = {'forget': [], 'input': [], 'output': []}

for t in range(seq_len):
    h, C, cache = lstm_step_forward(...)
    f, i, o = cache[3], cache[4], cache[6]
    gates['forget'].append(f)
    gates['input'].append(i)
    gates['output'].append(o)

# Plot
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.array(gates['forget']).T, cmap='RdYlGn', aspect='auto')
plt.title('Forget Gate (0=forget, 1=keep)')
plt.xlabel('Time step')
```

### 2. Cell State Evolution

```python
# Visualize cell state over time
cell_states = np.array(all_cell_states).T  # (hidden_size, seq_len)

plt.figure(figsize=(12, 6))
plt.imshow(cell_states, cmap='RdBu', aspect='auto')
plt.colorbar(label='Cell state value')
plt.xlabel('Time step')
plt.ylabel('Hidden unit')
plt.title('LSTM Cell State Evolution')
```

### 3. Gradient Flow Comparison

```python
# Compare gradient norms: LSTM vs vanilla RNN
lstm_grads = []  # Collect during LSTM backprop
rnn_grads = []   # Collect during RNN backprop

plt.figure(figsize=(10, 6))
plt.plot(lstm_grads, label='LSTM', linewidth=2)
plt.plot(rnn_grads, label='Vanilla RNN', linewidth=2, alpha=0.7)
plt.yscale('log')
plt.xlabel('Time step (backward)')
plt.ylabel('Gradient norm')
plt.title('Gradient Flow: LSTM vs RNN')
plt.legend()
```

---

## 🏋️ Exercises

### Exercise 1: Build LSTM from Scratch (⏱️⏱️⏱️)
Implement a complete LSTM in NumPy. Compare training curves with Day 1's vanilla RNN on the same data.

### Exercise 2: Gate Analysis (⏱️⏱️)
Train an LSTM on text, then visualize gate activations. Can you see patterns? (e.g., forget gate activating at sentence boundaries?)

### Exercise 3: Ablation Study (⏱️⏱️)
Remove one gate at a time:
- No forget gate (always remember everything)
- No input gate (always add new info)
- No output gate (always output everything)

Which one hurts performance most?

### Exercise 4: Long-Range Dependencies (⏱️⏱️)
Create a synthetic task: "Remember the first character of the sequence and output it at the end." Show that LSTM succeeds where vanilla RNN fails.

### Exercise 5: GRU Comparison (⏱️⏱️⏱️)
Implement a GRU (Gated Recurrent Unit) - LSTM's simpler cousin with only 2 gates. Compare performance and training speed.

---

## 🚀 Going Further

### LSTM Variants

1. **Peephole Connections** (Gers & Schmidhuber, 2000)
   - Gates can "peek" at cell state
   - Slightly better on some tasks

2. **GRU** (Cho et al., 2014)
   - 2 gates instead of 3
   - Faster, often comparable performance
   - Simpler = less parameters

3. **Bidirectional LSTM**
   - Process sequence forward AND backward
   - Used in BERT, ELMo

### When to Use LSTMs Today?

**Still useful for:**
- ✅ Time series with < 500 steps
- ✅ Streaming data (online learning)
- ✅ Resource-constrained environments
- ✅ When you need interpretability (gate analysis)

**Transformers better for:**
- ❌ Very long sequences (1000+ tokens)
- ❌ Parallelizable training
- ❌ Large-scale language models
- ❌ Attention to specific positions

---

## 📚 Resources

### Must-Read
- 📖 [Original blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah
- 📄 [LSTM paper](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber (1997)
- 📄 [Learning to Forget](https://ieeexplore.ieee.org/document/818041) - Gers et al. (2000)

### Visualizations
- 🎥 [LSTM visualization](https://github.com/HariWu1995/LSTM-Visualizer) - Interactive gates
- 📊 [Distill.pub on attention](https://distill.pub/2016/augmented-rnns/) - Advanced LSTM variants

### Implementations
- 💻 [Karpathy's char-rnn](https://github.com/karpathy/char-rnn) - Lua/Torch (historical)
- 💻 [PyTorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) - Production use

---

## 🎓 Key Takeaways

1. **LSTMs solve vanishing gradients** through a separate cell state with gated connections
2. **Three gates control information flow**: forget, input, output
3. **The cell state is a "highway"** for gradients to flow backward through time
4. **Gates are learned**, not hand-crafted - the network decides what to remember
5. **Still relevant today** despite transformers, especially for specific use cases

---

**Completed Day 2?** Move on to **[Day 3: Neural Machine Translation](../03_Neural_Machine_Translation/)** where we'll use LSTMs for translation!

**Questions?** Open an issue or check the [exercises](exercises/) for hands-on practice.

---

*"The key thing to understand about LSTMs is that the cell state is a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions."* - Christopher Olah
