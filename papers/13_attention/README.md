# Day 13: Attention Is All You Need (The Transformer)

> *"Attention Is All You Need"* - Vaswani et al. (2017)

**📖 Original Paper:** https://arxiv.org/abs/1706.03762

**⏱️ Time to Complete:** 4-6 hours

**🎯 What You'll Learn:**
- Why attention replaced recurrence entirely
- The mechanics of scaled dot-product attention
- Multi-head attention and why parallel heads matter
- Positional encoding - teaching position without recurrence
- Building a complete Transformer from scratch
- Why this architecture "ate the world"

---

## 🧠 The Big Idea

**In one sentence:** Replace recurrence with attention - let every position attend to every other position directly, in parallel, solving the sequential bottleneck and vanishing gradient problems.

### The Problem with RNNs/LSTMs

Remember Day 2? LSTMs solved the vanishing gradient problem, but they still have a critical flaw:

**Sequential Processing** = Can't parallelize - training is slow.

```
word_1 → word_2 → word_3 → ... → word_n
```

Each step must wait for the previous one. On modern GPUs, this is a huge bottleneck. A 100-word sentence requires 100 sequential steps.

Also, even with LSTMs, long-range dependencies require information to traverse many steps, potentially weakening the signal.

### The Transformer Solution

Process all positions simultaneously:

```
[word_1, word_2, word_3, ..., word_n] → Self-Attention → [out_1, out_2, ..., out_n]
```

**Benefits:**
- ✅ **Parallel**: All positions computed at once - fast training on GPUs
- ✅ **Direct connections**: Any position can attend to any other in one step
- ✅ **No vanishing gradients**: Direct paths for gradients
- ✅ **Unlimited context**: Attention over entire sequence

---

## 🤔 Why "Attention Is All You Need"?

This isn't just another architecture—it's a **paradigm shift**:

**The core claim:** You don't need recurrence or convolution. Pure attention is sufficient.

Before Transformers:
- ❌ RNNs were slow to train (sequential)
- ❌ CNNs for sequences needed many layers for long-range
- ❌ Attention was used WITH RNNs (as in seq2seq + attention)

After Transformers:
- ✅ Machine translation training became 10x faster
- ✅ Larger models became feasible (scale is all you need!)
- ✅ BERT, GPT, T5, ViT - all based on this paper
- ✅ Foundation of modern AI (ChatGPT, Claude, Gemini, etc.)

This architecture became the basis for virtually every major language model since 2017.

---

## 🌍 Real-World Analogy

### The Meeting Room Analogy

**RNN Way (Sequential):**
Imagine a business meeting where people speak one at a time in order:
- Person 1 speaks, person 2 listens
- Person 2 speaks, person 3 listens
- By person 10, the message from person 1 is garbled

**Transformer Way (Parallel Attention):**
Everyone has a microphone and can hear everyone else:
- Person 1 asks a question, everyone responds
- Person 1 decides who to pay attention to (attention weights)
- Direct communication between person 1 and person 10!

### The Dictionary Lookup Analogy

Think of attention as a **soft dictionary lookup**:

```python
# Hard lookup (traditional)
value = dictionary[key]  # Exact match required

# Soft lookup (attention)
value = sum(similarity(query, key_i) * value_i for all i)
# Returns weighted combination based on similarity
```

The magic: queries, keys, and values are all **learned**!

---

## 📊 The Architecture

### Scaled Dot-Product Attention

The core computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I contain?
- **V (Value)**: What do I return if matched?

**Step-by-step:**

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)
    V: (batch, heads, seq_k, d_v)
    """
    d_k = K.shape[-1]
    
    # Step 1: Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # (batch, heads, seq_q, seq_k)
    
    # Step 2: Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (for decoder)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    
    # Step 4: Softmax to get weights (sum to 1)
    weights = softmax(scores, dim=-1)
    
    # Step 5: Weighted sum of values
    output = weights @ V
    
    return output
```

**Why scale by √d_k?**
- Dot products grow with dimension: variance = d_k
- Large values → softmax saturates → vanishing gradients
- Scaling keeps variance stable → smooth softmax → better training

### Multi-Head Attention

One attention head learns one type of relationship. But language has MANY:
- Syntactic (subject-verb agreement)
- Semantic (word meaning similarity)
- Positional (nearby words)
- Coreference (pronouns to nouns)

**Solution:** Run **h** attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```python
def multi_head_attention(Q, K, V, n_heads=8):
    d_model = Q.shape[-1]
    d_k = d_model // n_heads
    
    # Project to h heads
    Q = linear(Q, W_Q).reshape(batch, seq, n_heads, d_k).transpose(1, 2)
    K = linear(K, W_K).reshape(batch, seq, n_heads, d_k).transpose(1, 2)
    V = linear(V, W_V).reshape(batch, seq, n_heads, d_k).transpose(1, 2)
    
    # Attention per head
    attn = scaled_dot_product_attention(Q, K, V)
    
    # Concatenate and project
    output = attn.transpose(1, 2).reshape(batch, seq, d_model)
    return linear(output, W_O)
```

Each head sees the data differently and learns different patterns!

### Positional Encoding

Without recurrence, the model has no sense of order!

"The cat sat on the mat" = "mat the on sat cat The" (same to pure attention)

**Solution:** Add position information using sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions
    
    return pe
```

**Why sinusoids?**
- Unique encoding for each position
- Can extrapolate to longer sequences than training
- Relative positions computable as linear functions: PE(pos+k) = f(PE(pos))

### The Full Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       TRANSFORMER                           │
├─────────────────────────┬───────────────────────────────────┤
│       ENCODER           │           DECODER                 │
├─────────────────────────┼───────────────────────────────────┤
│  Input Embedding        │  Output Embedding (shifted right) │
│  + Positional Encoding  │  + Positional Encoding            │
│           ↓             │             ↓                     │
│  ┌─────────────────┐    │  ┌────────────────────────────┐   │
│  │ Self-Attention  │×N  │  │ Masked Self-Attention     │    │
│  │ Add & Norm      │    │  │ Add & Norm                │×N  │
│  │ Feed-Forward    │    │  │ Cross-Attention           │    │
│  │ Add & Norm      │    │  │ Add & Norm                │    │
│  └─────────────────┘    │  │ Feed-Forward              │    │
│           ↓             │  │ Add & Norm                │    │
│     Encoder Output ─────┼──→ └──────────────────────────────┘
│                         │             ↓                     │
│                         │      Linear + Softmax             │
│                         │             ↓                     │
│                         │         Output Probs              │
└─────────────────────────┴───────────────────────────────────┘
```

**Encoder Block:**
1. Multi-Head Self-Attention (attend to all input positions)
2. Add & Normalize (residual + layer norm)
3. Feed-Forward Network (2 linear layers with ReLU)
4. Add & Normalize

**Decoder Block:**
1. Masked Multi-Head Self-Attention (can't see future!)
2. Add & Normalize
3. Multi-Head Cross-Attention (attend to encoder output)
4. Add & Normalize
5. Feed-Forward Network
6. Add & Normalize

---

## 💡 The Key Innovations

### 1. Self-Attention Replaces Recurrence

The computational complexity:
- RNN: O(n) sequential operations
- Self-attention: O(1) sequential, but O(n²) parallel

On GPUs, parallel is MUCH faster than sequential!

### 2. Residual Connections (from ResNet!)

Every sub-layer has:
```python
output = layer_norm(x + sublayer(x))
```

This allows gradients to flow directly, enabling very deep models (6+ layers).

### 3. Layer Normalization

Unlike Batch Norm, Layer Norm:
- Works on single samples (no batch dependency)
- Normalizes across features, not batch
- Works well with variable-length sequences

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta
```

### 4. Causal Masking in Decoder

The decoder can't see future tokens (autoregressive):

```python
def create_causal_mask(seq_len):
    # Upper triangle = True = masked
    return np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
```

---

## 🔧 Implementation Details

### Feed-Forward Network

```python
def feed_forward(x, W1, b1, W2, b2):
    """Two linear layers with ReLU in between."""
    hidden = relu(x @ W1 + b1)  # d_model -> d_ff
    return hidden @ W2 + b2       # d_ff -> d_model
```

The hidden dimension is typically 4× the model dimension (d_ff = 4 * d_model).

### Learning Rate Schedule

The paper uses warmup + decay:

$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$

```python
def get_lr(step, d_model, warmup=4000):
    return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))
```

---

## 📊 Key Hyperparameters (Original Paper)

| Parameter | Base Model | Big Model |
|-----------|-----------|-----------|
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| n_heads | 8 | 16 |
| n_layers | 6 | 6 |
| d_k = d_v | 64 | 64 |
| Dropout | 0.1 | 0.3 |
| Parameters | 65M | 213M |

---

## 🎯 Training Tips

### 1. Label Smoothing
```python
# Instead of one-hot [0, 0, 1, 0, 0]
# Use [0.025, 0.025, 0.9, 0.025, 0.025]
smooth_labels = (1 - epsilon) * one_hot + epsilon / vocab_size
```

### 2. Dropout Placement
- After attention weights
- After each sub-layer (before adding residual)
- On embeddings

### 3. Weight Initialization
```python
# Xavier/Glorot for linear layers
W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))
```

---

## 📈 Visualizations

### Attention Patterns

Different heads learn different patterns:
- **Head 1**: Attend to previous word
- **Head 2**: Attend to syntactic head
- **Head 3**: Attend to rare words
- **Head 4**: Attend to punctuation

Run `visualization.py` to see attention heatmaps!

---

## 🏋️ Exercises

We have 5 progressive exercises:

1. **Scaled Dot-Product Attention** - The core computation
2. **Multi-Head Attention** - Parallel attention heads
3. **Positional Encoding** - Sinusoidal position embeddings
4. **Encoder Block** - Self-attention + FFN + residuals
5. **Full Transformer** - Encoder-decoder architecture

Start with `exercises/exercise_01_scaled_dot_product.py`!

---

## 🚀 Going Further

### Transformer Variants

1. **Encoder-only**: BERT (bidirectional, for understanding)
2. **Decoder-only**: GPT (autoregressive, for generation)
3. **Encoder-Decoder**: T5, BART (for seq2seq tasks)
4. **Vision Transformer**: ViT (patches as tokens)

### Modern Improvements

- **Relative position encodings** (more flexible than absolute)
- **Sparse attention** (for very long sequences)
- **Flash Attention** (memory-efficient GPU implementation)
- **Mixture of Experts** (scale efficiently)

---

## 📚 Resources

### Must-Read
- 📖 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- 📖 [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- 📄 [Original Paper](https://arxiv.org/abs/1706.03762) - Vaswani et al.

### Implementations
- 💻 [HuggingFace Transformers](https://github.com/huggingface/transformers)
- 💻 [PyTorch nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

---

## 🎓 Key Takeaways

1. **Attention replaces recurrence** - parallel processing is key
2. **Scaling by √d_k** prevents softmax saturation
3. **Multi-head attention** captures different relationship types
4. **Positional encoding** provides order information
5. **Residual connections + layer norm** enable deep models
6. **Encoder-decoder** for seq2seq, decoder-only for generation

---

**Completed Day 13?** Move on to **Day 14** or explore transformer variants!

**Questions?** Check the [exercises](exercises/) or the [notebook](notebook.ipynb).

---

*"Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."* - Vaswani et al.
