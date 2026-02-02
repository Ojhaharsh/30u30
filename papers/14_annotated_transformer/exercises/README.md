# Day 14 Exercises: The Annotated Transformer

> Build production-quality Transformer components from scratch in PyTorch

---

## 🎯 Overview

These exercises take you from individual components to a complete, trainable Transformer. Each exercise builds on the previous one, culminating in a working translation model.

**Prerequisites:**
- Day 13 completed (Transformer theory)
- PyTorch basics (tensors, nn.Module, autograd)
- Understanding of attention mechanism

---

## 📚 Exercise Progression

### Exercise 1: Attention with Masking
**Difficulty:** ⭐⭐ (Foundational)

Implement the core attention function with proper masking support.

**You'll learn:**
- How scaled dot-product attention works in PyTorch
- Why masking uses `-1e9` instead of `-inf`
- How to handle the math of attention in batch

**Key functions:**
- `attention(query, key, value, mask, dropout)`
- `subsequent_mask(size)`

---

### Exercise 2: Multi-Head Attention Module
**Difficulty:** ⭐⭐⭐ (Intermediate)

Build the complete multi-head attention as an `nn.Module`.

**You'll learn:**
- How to split embeddings across heads (view + transpose)
- Why we need 4 linear projections (Q, K, V, output)
- How to concatenate heads back together

**Key class:**
- `MultiHeadedAttention(nn.Module)`

---

### Exercise 3: Encoder Stack
**Difficulty:** ⭐⭐⭐ (Intermediate)

Compose encoder layers with residual connections and layer norm.

**You'll learn:**
- Pre-norm vs post-norm architecture
- How residual connections enable deep networks
- Proper layer stacking with `clones()`

**Key classes:**
- `EncoderLayer(nn.Module)`
- `Encoder(nn.Module)`
- `SublayerConnection(nn.Module)`

---

### Exercise 4: Complete Training Pipeline
**Difficulty:** ⭐⭐⭐⭐ (Advanced)

Build the training infrastructure: batching, loss, optimization.

**You'll learn:**
- How to create training batches with proper masking
- Label smoothing and why it helps
- The Noam learning rate schedule

**Key classes:**
- `Batch` (data holder)
- `LabelSmoothing(nn.Module)`
- `NoamOpt` (optimizer wrapper)

---

### Exercise 5: Inference and Decoding
**Difficulty:** ⭐⭐⭐⭐ (Advanced)

Implement greedy and beam search decoding.

**You'll learn:**
- How autoregressive generation works
- Why encoder output is computed once and reused
- Trade-offs between greedy and beam search

**Key functions:**
- `greedy_decode(model, src, src_mask, max_len, start_symbol)`
- `beam_search_decode(model, src, src_mask, max_len, beam_size)`

---

## 🚀 Getting Started

```bash
# Navigate to exercises
cd papers/14_annotated_transformer/exercises

# Start with Exercise 1
python exercise_01_attention.py

# Check your solution
python solutions/solution_01_attention.py
```

---

## 💡 Tips for Success

### General Advice

1. **Print shapes obsessively** - Most bugs are shape mismatches
2. **Start with small dimensions** - Use d_model=8, h=2 for debugging
3. **Visualize attention weights** - Helps verify correctness
4. **Compare with reference** - Check against `implementation.py`

### Common Pitfalls

| Problem | Solution |
|---------|----------|
| Mask broadcasting | Use `.unsqueeze()` to add dimensions |
| Gradient explosion | Check initialization and learning rate |
| Memory errors | Reduce batch size or use gradient checkpointing |
| Wrong attention pattern | Verify mask creation and application |

---

## 📁 File Structure

```
exercises/
├── README.md                 # This file
├── exercise_01_attention.py  # Attention with masking
├── exercise_02_multihead.py  # Multi-head attention module
├── exercise_03_encoder.py    # Encoder stack
├── exercise_04_training.py   # Training pipeline
├── exercise_05_inference.py  # Decoding strategies
└── solutions/
    ├── solution_01_attention.py
    ├── solution_02_multihead.py
    ├── solution_03_encoder.py
    ├── solution_04_training.py
    └── solution_05_inference.py
```

---

## 🎓 Learning Objectives

By completing these exercises, you will:

1. ✅ Implement attention from scratch in PyTorch
2. ✅ Understand view/transpose for multi-head splitting
3. ✅ Build residual connections with proper normalization
4. ✅ Create training batches with complex masking
5. ✅ Implement label smoothing regularization
6. ✅ Use warmup learning rate schedules
7. ✅ Perform autoregressive inference
8. ✅ Compare greedy vs beam search decoding

---

## 🔗 Connection to Day 13

| Day 13 (NumPy) | Day 14 (PyTorch) |
|----------------|------------------|
| Manual backprop understanding | Autograd handles it |
| Single example | Batched processing |
| CPU only | GPU accelerated |
| Educational | Production-ready |

---

**Ready to start?** Open `exercise_01_attention.py` and let's build! 🚀
