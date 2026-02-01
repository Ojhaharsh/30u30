# Day 13: Exercises - Attention Is All You Need

Master the Transformer architecture through 5 progressive exercises!

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Scaled Dot-Product Attention | ⭐ | 30 min |
| 2 | Multi-Head Attention | ⭐⭐ | 45 min |
| 3 | Positional Encoding | ⭐⭐ | 30 min |
| 4 | Encoder Block | ⭐⭐⭐ | 60 min |
| 5 | Full Transformer | ⭐⭐⭐⭐ | 90 min |

---

## Exercise 1: Scaled Dot-Product Attention ⭐

**File:** `exercise_01_scaled_dot_product.py`

The core computation of the Transformer:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Your Tasks:**
1. Implement the attention function
2. Handle optional masking
3. Verify attention weights sum to 1

**Key Concepts:**
- Query, Key, Value interpretation
- Why we scale by sqrt(d_k)
- Attention as soft lookup

---

## Exercise 2: Multi-Head Attention ⭐⭐

**File:** `exercise_02_multi_head_attention.py`

Multiple attention heads capture different relationships:

**Your Tasks:**
1. Project Q, K, V into multiple heads
2. Apply attention to each head in parallel
3. Concatenate and project back

**Key Concepts:**
- Different heads learn different patterns
- Parallel processing across heads
- Dimension splitting: d_k = d_model / n_heads

---

## Exercise 3: Positional Encoding ⭐⭐

**File:** `exercise_03_positional_encoding.py`

Without recurrence, we need explicit position information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Your Tasks:**
1. Implement sinusoidal positional encoding
2. Visualize the encoding matrix
3. Explore position similarity

**Key Concepts:**
- Why sinusoids? (relative positions as linear functions)
- Generalization to longer sequences
- Adding vs concatenating

---

## Exercise 4: Encoder Block ⭐⭐⭐

**File:** `exercise_04_encoder_block.py`

A complete encoder layer:

```
x → Self-Attention → Add & Norm → FFN → Add & Norm → output
```

**Your Tasks:**
1. Combine multi-head attention with residual
2. Add layer normalization
3. Implement feed-forward network
4. Stack into a complete encoder

**Key Concepts:**
- Residual connections for gradient flow
- Layer norm vs batch norm
- Feed-forward: expand then contract

---

## Exercise 5: Full Transformer ⭐⭐⭐⭐

**File:** `exercise_05_full_transformer.py`

The complete encoder-decoder architecture:

**Your Tasks:**
1. Build the encoder stack
2. Build the decoder stack (with masking!)
3. Connect encoder to decoder via cross-attention
4. Add embeddings and output projection
5. Train on a simple sequence task

**Key Concepts:**
- Causal masking in decoder
- Cross-attention: decoder queries encoder
- Teacher forcing during training

---

## How to Run

```bash
# Start with Exercise 1
python exercise_01_scaled_dot_product.py

# Progress through each exercise
python exercise_02_multi_head_attention.py
python exercise_03_positional_encoding.py
python exercise_04_encoder_block.py
python exercise_05_full_transformer.py

# Check your solutions
cd solutions/
python solution_01_scaled_dot_product.py
```

---

## Tips

1. **Understand shapes**: Attention is all about tensor dimensions!
   - (batch, seq_len, d_model) is your friend
   - Multi-head reshapes to (batch, heads, seq_len, d_k)

2. **Test incrementally**: Each function should work before combining

3. **Visualize attention**: Plot attention weights to debug

4. **Start simple**: Use small dimensions (d_model=32) to debug

5. **Compare to reference**: Check against `implementation.py`

---

## Expected Learning Outcomes

After completing these exercises, you should:

✅ Understand how attention replaces recurrence
✅ Know why scaling by sqrt(d_k) is crucial
✅ Grasp multi-head attention's parallel perspectives
✅ Implement positional encoding from scratch
✅ Build encoder and decoder blocks
✅ Train a working Transformer on simple tasks

---

## Bonus Challenges

1. **Add dropout**: Implement dropout in attention and FFN
2. **Pre-norm vs Post-norm**: Try both layer norm orderings
3. **Relative positions**: Replace absolute with relative encoding
4. **Beam search**: Implement beam search decoding
5. **Visualization**: Create attention heatmaps for analysis

Good luck! The Transformer awaits. 🤖
