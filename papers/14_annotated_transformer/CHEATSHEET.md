# The Annotated Transformer - Quick Reference Cheatsheet

> Production PyTorch Transformer implementation at a glance

---

## 🏗️ Module Hierarchy

```python
EncoderDecoder(
    encoder=Encoder(EncoderLayer × N),
    decoder=Decoder(DecoderLayer × N),
    src_embed=Embeddings + PositionalEncoding,
    tgt_embed=Embeddings + PositionalEncoding,
    generator=Linear + Softmax
)
```

---

## 📦 Core Classes

### Attention Function

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

### Multi-Head Attention

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # Project and reshape: (batch, seq, d_model) → (batch, h, seq, d_k)
        q, k, v = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for lin, x in zip(self.linears, (query, key, value))]
        x, _ = attention(q, k, v, mask=mask, dropout=self.dropout)
        # Concat: (batch, h, seq, d_k) → (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

### Feed-Forward Network

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

### Sublayer Connection (Residual + Norm)

```python
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Pre-norm!
```

### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
```

---

## 🎭 Masking

### Padding Mask (Source)

```python
src_mask = (src != pad_idx).unsqueeze(-2)
# Shape: (batch, 1, seq)
```

### Subsequent Mask (Decoder Self-Attention)

```python
def subsequent_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1) == 0
    return mask
# Result:
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]
```

### Combined Target Mask

```python
tgt_mask = (tgt != pad_idx).unsqueeze(-2) & subsequent_mask(tgt.size(-1))
```

---

## 📊 Training Components

### Batch Object

```python
class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]      # Decoder input
            self.tgt_y = tgt[:, 1:]     # Target for loss
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()
```

### Label Smoothing

```python
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    def forward(self, x, target):
        true_dist = x.clone().fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        return self.criterion(x, true_dist)
```

### Noam Learning Rate Schedule

```python
def rate(step, d_model, warmup=4000):
    if step == 0: step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
```

---

## 🔄 Forward Pass Flow

```
Input Tokens
     ↓
Embedding × √d_model
     ↓
+ Positional Encoding
     ↓
┌────────────────────────┐
│    Encoder Layer × N   │
│  ┌──────────────────┐  │
│  │ Self-Attention   │  │
│  │ + Residual + Norm│  │
│  └──────────────────┘  │
│  ┌──────────────────┐  │
│  │ Feed-Forward     │  │
│  │ + Residual + Norm│  │
│  └──────────────────┘  │
└────────────────────────┘
     ↓
Encoder Output (memory)
     ↓
┌────────────────────────┐
│    Decoder Layer × N   │
│  ┌──────────────────┐  │
│  │ Masked Self-Attn │  │
│  │ + Residual + Norm│  │
│  └──────────────────┘  │
│  ┌──────────────────┐  │
│  │ Cross-Attention  │←─┼── Encoder Output
│  │ + Residual + Norm│  │
│  └──────────────────┘  │
│  ┌──────────────────┐  │
│  │ Feed-Forward     │  │
│  │ + Residual + Norm│  │
│  └──────────────────┘  │
└────────────────────────┘
     ↓
Linear + Log Softmax
     ↓
Output Probabilities
```

---

## 🎯 Inference

### Greedy Decoding

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long()
    
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1))
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
    return ys
```

---

## 📐 Shape Transformations

```python
# Multi-head attention shapes
query:  (batch, seq_q, d_model)
    ↓ linear projection
        (batch, seq_q, d_model)
    ↓ view + transpose
        (batch, h, seq_q, d_k)      # d_k = d_model // h

# Attention computation
scores: (batch, h, seq_q, seq_k)    # Q @ K.T
attn:   (batch, h, seq_q, seq_k)    # After softmax
output: (batch, h, seq_q, d_v)      # attn @ V

# Back to original shape
        (batch, seq_q, d_model)     # transpose + view + linear
```

---

## ⚙️ Hyperparameters (Base Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 512 | Model dimension |
| `d_ff` | 2048 | FFN hidden dimension (4×) |
| `h` | 8 | Number of attention heads |
| `d_k = d_v` | 64 | Per-head dimension |
| `N` | 6 | Number of layers (encoder & decoder) |
| `dropout` | 0.1 | Dropout rate |
| `warmup` | 4000 | LR warmup steps |
| `label_smoothing` | 0.1 | Smoothing factor |

---

## 🛠️ Utility Functions

### Clone Layers

```python
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

### Layer Norm

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### Embeddings with Scaling

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # Scale!
```

---

## 🚨 Common Gotchas

| Issue | Solution |
|-------|----------|
| Mask dimension error | Add `.unsqueeze(1)` for broadcast over heads |
| NaN in attention | Check for all-masked rows, use `-1e9` not `-inf` |
| Memory issues | Use gradient checkpointing or smaller batch |
| Slow training | Mixed precision with `torch.cuda.amp` |
| Embedding scale | Multiply by `√d_model` after lookup |

---

## 📝 Quick Model Creation

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, MultiHeadedAttention(h, d_model),
                             PositionwiseFeedForward(d_model, d_ff), dropout), N),
        Decoder(DecoderLayer(d_model, MultiHeadedAttention(h, d_model),
                             MultiHeadedAttention(h, d_model),
                             PositionwiseFeedForward(d_model, d_ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, dropout)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(d_model, dropout)),
        Generator(d_model, tgt_vocab)
    )
    # Xavier init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
```

---

*Keep this cheatsheet handy while implementing!* 📋
