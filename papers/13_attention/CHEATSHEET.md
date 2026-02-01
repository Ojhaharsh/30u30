# Day 13: Transformer Cheatsheet

## Core Idea
Replace recurrence with self-attention. Every position attends to every other position directly and in parallel.

---

## Quick Reference

### Scaled Dot-Product Attention
```python
def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights = softmax(scores, dim=-1)
    return weights @ V
```

### Multi-Head Attention
```python
def multi_head_attention(x, n_heads=8):
    d_model = x.shape[-1]
    d_k = d_model // n_heads
    
    # Project to Q, K, V
    Q = x @ W_Q  # (batch, seq, d_model)
    K = x @ W_K
    V = x @ W_V
    
    # Split into heads: (batch, heads, seq, d_k)
    Q = Q.reshape(batch, seq, n_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq, n_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq, n_heads, d_k).transpose(1, 2)
    
    # Attention per head
    attn = attention(Q, K, V)
    
    # Concatenate heads: (batch, seq, d_model)
    out = attn.transpose(1, 2).reshape(batch, seq, d_model)
    return out @ W_O
```

### Positional Encoding
```python
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
```

### Encoder Block
```python
def encoder_block(x):
    # Self-attention
    attn = multi_head_attention(x, x, x)
    x = layer_norm(x + dropout(attn))
    
    # Feed-forward
    ff = feed_forward(x)
    x = layer_norm(x + dropout(ff))
    return x
```

### Decoder Block
```python
def decoder_block(x, encoder_output, mask):
    # Masked self-attention
    self_attn = multi_head_attention(x, x, x, mask=mask)
    x = layer_norm(x + dropout(self_attn))
    
    # Cross-attention (Q from decoder, K/V from encoder)
    cross_attn = multi_head_attention(x, encoder_output, encoder_output)
    x = layer_norm(x + dropout(cross_attn))
    
    # Feed-forward
    ff = feed_forward(x)
    x = layer_norm(x + dropout(ff))
    return x
```

---

## Key Formulas

### Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Multi-Head
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## Hyperparameters (Paper)

| Param | Base | Big | Description |
|-------|------|-----|-------------|
| d_model | 512 | 1024 | Model dimension |
| d_ff | 2048 | 4096 | Feed-forward hidden dim |
| n_heads | 8 | 16 | Attention heads |
| n_layers | 6 | 6 | Encoder/decoder layers |
| d_k, d_v | 64 | 64 | Key/value dimension |
| dropout | 0.1 | 0.3 | Dropout rate |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       TRANSFORMER                           │
├─────────────────────────┬───────────────────────────────────┤
│       ENCODER           │           DECODER                 │
├─────────────────────────┼───────────────────────────────────┤
│  Input Embedding        │  Output Embedding (shifted right) │
│  + Positional Encoding  │  + Positional Encoding           │
│           ↓             │             ↓                     │
│  ┌─────────────────┐    │  ┌────────────────────────────┐  │
│  │ Self-Attention  │×N  │  │ Masked Self-Attention     │  │
│  │ Add & Norm      │    │  │ Add & Norm                │×N │
│  │ Feed-Forward    │    │  │ Cross-Attention           │  │
│  │ Add & Norm      │    │  │ Add & Norm                │  │
│  └─────────────────┘    │  │ Feed-Forward              │  │
│           ↓             │  │ Add & Norm                │  │
│     Encoder Output ─────┼──→ └────────────────────────────┘  │
│                         │             ↓                     │
│                         │      Linear + Softmax             │
│                         │             ↓                     │
│                         │         Output Probs              │
└─────────────────────────┴───────────────────────────────────┘
```

---

## Common Patterns

### Causal Mask (for Decoder)
```python
def create_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
```

### Padding Mask
```python
def create_padding_mask(seq, pad_token=0):
    return (seq == pad_token)
```

### Learning Rate Schedule
```python
def get_lr(step, d_model, warmup=4000):
    return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| NaN gradients | Large attention scores | Increase scaling factor |
| Slow convergence | No warmup | Add learning rate warmup |
| Attention collapse | All heads same | Add dropout to attention |
| Position ignored | Wrong PE dims | Check PE matches embedding dim |
| Memory OOM | Long sequences | Use gradient checkpointing |

---

## Variants

| Model | Type | Changes |
|-------|------|---------|
| BERT | Encoder only | Bidirectional, MLM pretraining |
| GPT | Decoder only | Causal, autoregressive |
| T5 | Encoder-Decoder | Relative position, text-to-text |
| ViT | Vision | Patch embeddings |

---

## Implementation Tips

1. **Initialize weights carefully**: Use Xavier/Glorot for projections
2. **Residual connections**: Essential for training deep models
3. **Layer norm**: Apply PRE-norm (before attention) in modern models
4. **Dropout placement**: After attention weights, after FFN
5. **Gradient clipping**: Use max norm ~1.0

---

## Quick Test

```python
# Minimal attention test
Q = np.random.randn(2, 4, 64)  # (batch, seq, d_k)
K = np.random.randn(2, 4, 64)
V = np.random.randn(2, 4, 64)

scores = Q @ K.transpose(0, 2, 1) / np.sqrt(64)
weights = softmax(scores, axis=-1)
output = weights @ V

assert output.shape == (2, 4, 64)
print("Attention works!")
```
