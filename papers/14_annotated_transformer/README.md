# Day 14: The Annotated Transformer

> *"The Annotated Transformer"* - Alexander Rush (2018)

**📖 Original Article:** https://nlp.seas.harvard.edu/annotated-transformer/

**⏱️ Time to Complete:** 4-5 hours

**🎯 What You'll Learn:**
- Production-quality PyTorch Transformer implementation
- Line-by-line code understanding with explanations
- Training loops, batching, and real-world considerations
- Label smoothing, learning rate scheduling, and optimization tricks
- Multi-GPU training and inference strategies

---

## 🧠 The Big Idea

**In one sentence:** Take the "Attention Is All You Need" paper and turn every equation into working PyTorch code, with explanations for every design decision.

### Why This Matters

Day 13 gave you the theory. Day 14 gives you **the code that actually runs**.

The Annotated Transformer is not a new architecture—it's the definitive implementation guide. Think of it as the difference between reading a recipe and watching a master chef cook it step by step.

```
Day 13 (Theory):  Attention(Q,K,V) = softmax(QK^T/√d_k)V
                           ↓
Day 14 (Code):    class MultiHeadedAttention(nn.Module):
                      def forward(self, query, key, value, mask=None):
                          # 40 lines of production PyTorch
```

---

## 🔄 Day 13 vs Day 14: Theory → Practice

| Aspect | Day 13 (Paper) | Day 14 (Annotated) |
|--------|----------------|-------------------|
| Focus | Mathematical foundations | Working PyTorch code |
| Attention | Formula + intuition | `nn.Module` implementation |
| Training | Conceptual | Actual training loop |
| Batching | Not covered | Masked batch handling |
| Examples | Synthetic | Real translation task |

**The gap this fills:** You understood the math. Now you'll understand why the code looks the way it does.

---

## 🏗️ Architecture Implementation

### The Module Hierarchy

```
Transformer
├── Encoder
│   ├── EncoderLayer (×N)
│   │   ├── MultiHeadedAttention (self)
│   │   ├── LayerNorm
│   │   ├── PositionwiseFeedForward
│   │   └── LayerNorm
│   └── LayerNorm (final)
├── Decoder
│   ├── DecoderLayer (×N)
│   │   ├── MultiHeadedAttention (self, masked)
│   │   ├── LayerNorm
│   │   ├── MultiHeadedAttention (cross)
│   │   ├── LayerNorm
│   │   ├── PositionwiseFeedForward
│   │   └── LayerNorm
│   └── LayerNorm (final)
├── Embeddings (source + target)
├── PositionalEncoding
└── Generator (final linear + softmax)
```

### Core Classes

#### 1. Multi-Head Attention

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask for all heads
        nbatches = query.size(0)
        
        # 1) Project: d_model => h × d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) Concat and project back
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

**Key insight:** The view/transpose operations split the embedding into `h` heads, each seeing `d_k` dimensions. This is just reshaping—no new parameters!

#### 2. Position-wise Feed-Forward

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

**Why d_ff = 4 × d_model?** The intermediate expansion gives the network more capacity to learn complex transformations before projecting back down.

#### 3. Residual Connection + Layer Norm

```python
class SublayerConnection(nn.Module):
    """Residual connection followed by layer norm."""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Note: norm BEFORE sublayer (pre-norm, like ResNet v2)
        return x + self.dropout(sublayer(self.norm(x)))
```

**Critical detail:** The Annotated Transformer uses pre-norm (normalize then apply sublayer), which differs slightly from the original paper's post-norm. Pre-norm trains more stably.

#### 4. Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

**Why register_buffer?** Positional encodings are fixed (not learned), but they should move to GPU with the model. `register_buffer` handles this.

---

## 🎯 Training Infrastructure

### The Batch Object

Real translation requires careful batching:

```python
class Batch:
    """Hold a batch of data with mask for training."""
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # Input to decoder
            self.tgt_y = tgt[:, 1:]  # Target for loss
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum().item()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask
```

**The trick:** Target is shifted—decoder input is `[<start>, tok1, tok2]`, decoder target is `[tok1, tok2, <end>]`. This teaches the model to predict the next token.

### Label Smoothing

Don't use pure one-hot targets:

```python
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
    
    def forward(self, x, target):
        # x: (batch * seq, vocab)
        # target: (batch * seq,)
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target == self.padding_idx)
        true_dist[mask] = 0
        return self.criterion(x, true_dist)
```

**Why?** One-hot says "100% sure it's this word." Label smoothing says "90% sure, 10% spread across other words." This prevents overconfidence and improves generalization.

### Learning Rate Schedule

The famous warmup schedule:

```python
def rate(step, d_model, warmup=4000):
    """Noam learning rate schedule."""
    if step == 0:
        step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
```

```
LR
 │    /\
 │   /  \____
 │  /        \_____
 │ /                \____
 └──────────────────────── step
   warmup    decay
```

**Why warmup?** Early in training, the model is random. Large learning rates would cause instability. Warmup gradually increases LR, then decays.

---

## 🔄 Training Loop

```python
def run_epoch(data_iter, model, loss_compute):
    total_tokens = 0
    total_loss = 0
    
    for batch in data_iter:
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
    
    return total_loss / total_tokens
```

### Inference: Greedy Decoding

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask,
            ys, subsequent_mask(ys.size(1)).type_as(src)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)
        
        if next_word.item() == end_symbol:
            break
    
    return ys
```

**How it works:**
1. Encode source once (reused for all steps)
2. Start with `<start>` token
3. Predict next token, append to sequence
4. Repeat until `<end>` or max length

---

## 🛠️ Implementation Details That Matter

### 1. Clone Function

```python
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

**Why deepcopy?** Each layer needs its own parameters. Shallow copy would share weights.

### 2. Subsequent Mask

```python
def subsequent_mask(size):
    """Mask out subsequent positions (for decoder)."""
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0
```

Creates:
```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

Position i can only attend to positions ≤ i.

### 3. Attention Function

```python
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = scores.softmax(dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn
```

**Why -1e9 instead of -inf?** Some numerical stability. After softmax, e^(-1e9) ≈ 0.

---

## 📊 Complete Model Assembly

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model
```

---

## 🎓 Key Takeaways

### From Math to Code

| Paper Concept | Code Implementation |
|---------------|---------------------|
| Q, K, V projection | `nn.Linear(d_model, d_model)` × 3 |
| Multi-head split | `.view(batch, seq, h, d_k).transpose(1, 2)` |
| Scaled dot-product | `matmul(Q, K.T) / sqrt(d_k)` |
| Masking | `masked_fill(mask == 0, -1e9)` |
| Residual connection | `x + sublayer(x)` |
| Layer norm | `nn.LayerNorm(d_model)` |

### Production Considerations

1. **Pre-norm vs Post-norm**: Pre-norm (norm before sublayer) trains more stably
2. **Weight initialization**: Xavier uniform for matrices, zeros for biases
3. **Dropout placement**: After attention weights, after sublayers, on embeddings
4. **Gradient accumulation**: For large batches on limited GPU memory
5. **Mixed precision**: `torch.cuda.amp` for faster training

---

## 🏋️ Exercises

We have 5 progressive exercises:

1. **Attention Implementation** - Build attention with proper masking
2. **Encoder Stack** - Compose encoder layers with residuals
3. **Full Transformer** - Wire up encoder-decoder architecture
4. **Training Loop** - Implement batching, loss, and optimization
5. **Inference** - Greedy and beam search decoding

Start with `exercises/exercise_01_attention.py`!

---

## 📚 Resources

### Essential Reading
- 📖 [The Annotated Transformer (2022 update)](https://nlp.seas.harvard.edu/annotated-transformer/) - Rush et al.
- 📖 [Original Paper: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 📖 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### Reference Implementations
- 💻 [Harvard NLP Code](https://github.com/harvardnlp/annotated-transformer)
- 💻 [PyTorch nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- 💻 [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

## 🔗 Connection to Day 13

| Day 13 | Day 14 |
|--------|--------|
| "What is attention?" | "How do I code attention?" |
| Mathematical intuition | PyTorch implementation |
| Architecture diagrams | Module hierarchy |
| Training concepts | Actual training loop |
| Synthetic examples | Real translation task |

**Together:** Complete understanding of both theory and practice.

---

**Completed Day 14?** You now have production-ready Transformer code. Move on to **Day 15** to explore BERT/GPT variants!

---

*"The goal of this post is to provide a complete and working implementation of the Transformer architecture. The code is annotated line-by-line."* - Alexander Rush
