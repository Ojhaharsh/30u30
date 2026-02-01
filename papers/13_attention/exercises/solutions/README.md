# Day 13: Exercise Solutions

Complete solutions with detailed explanations.

## Key Concepts Summary

### 1. Scaled Dot-Product Attention
The core formula: `softmax(QK^T / sqrt(d_k)) @ V`
- **Scaling by sqrt(d_k)**: Prevents dot products from growing too large
- **Masking**: Set masked positions to -infinity before softmax
- **Soft lookup**: Attention is differentiable dictionary access

### 2. Multi-Head Attention
- **Parallel heads**: Each head learns different relationships
- **Reshape trick**: Split d_model into (n_heads, d_k)
- **Concatenation**: Merge heads back before output projection

### 3. Positional Encoding
- **Sinusoids**: sin/cos at different frequencies
- **Formula**: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- **Benefits**: Unique positions, extrapolates to longer sequences

### 4. Encoder Block
- **Self-attention**: Every position attends to every other
- **Residual + Norm**: For stable gradients
- **Feed-forward**: Independent per-position transformation

### 5. Full Transformer
- **Encoder**: Stack of encoder blocks
- **Decoder**: Masked self-attention + cross-attention + FFN
- **Causal mask**: Prevent decoder from seeing future

## Running Solutions

```bash
python solution_01_scaled_dot_product.py
python solution_02_multi_head_attention.py
python solution_03_positional_encoding.py
python solution_04_encoder_block.py
python solution_05_full_transformer.py
```

## Common Pitfalls

1. **Forgetting the scale factor**: Divide by sqrt(d_k)!
2. **Wrong transpose axes**: Multi-head reshape needs careful axis ordering
3. **Missing causal mask**: Decoder must not see future tokens
4. **Position broadcasting**: PE adds to (batch, seq, d_model)
