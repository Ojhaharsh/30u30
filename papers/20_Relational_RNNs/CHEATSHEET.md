# Day 20 Cheat Sheet: Relational Recurrent Neural Networks

## The Big Idea (30 seconds)

**Problem:** Standard RNNs (LSTMs) squash all history into a single vector, causing "relational amnesia" over long sequences.
**Insight:** Instead of one vector, keep a "team" of memory slots. Let them "talk" to each other (attend) at every step before updating.
**Result:** A model that can reason about objects and relationships over time (e.g., "The ball I saw at t=5 is the same one at t=100").

**The one picture to remember:**

```text
       Input x_t
          |
    +-----+-----+
    | Project & |  <-- Concatenate input to each slot
    |   Tile    |
    +-----+-----+
          |
    [ Slot 1 ] [ Slot 2 ] ... [ Slot N ]  (Previous Memory M_{t-1})
          |        |              |
    +---------------------------------+
    |   Multi-Head Self-Attention     |  <-- "Team Meeting"
    +---------------------------------+
          |        |              |
       [Attended Information]
          |        |              |
    +---------------------------------+
    |    Gated Update (LSTM-style)    |  <-- "Write/Forget"
    +---------------------------------+
          |
    New Memory M_t
```

---

## Quick Start

```bash
cd papers/20_Relational_RNNs

# 1. Train RMC vs LSTM (The "Showdown")
python train_minimal.py --model both --save_plot

# 2. Visualize Attention (The "Thought Process")
python train_minimal.py --model rmc --visualize

# 3. Step-by-step Implementation Guide
python exercises/exercise_01_attention.py
```

---

## Key Parameters for RMC

| Parameter | Typical Range | What It Does | Tips |
|-----------|--------------|--------------|------|
| `num_slots` | 4-16 | Number of memory vectors | More slots = more "objects" tracked. Too many = slow. |
| `slot_size` | 32-128 | Size of each vector | Total memory = slots * size. |
| `num_heads` | 4-8 | Attention heads | Allows slots to attend to different *types* of info simultaneously. |
| `num_blocks` | 1 | Stacked RMC layers | usually 1 is enough. |
| `gate_style` | LSTM | Input/Forget gates | **Crucial:** Initialize forget bias to 1.0! |

---

## The Math (Copy-Paste Ready)

### 1. Multi-Head Attention (MHDPA)

```python
def attention(query, key, value):
    """
    Standard Scaled Dot-Product Attention.
    Args:
        query: [batch, heads, slots, d_k]
        key:   [batch, heads, slots, d_k]
        value: [batch, heads, slots, d_v]
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, value), probs
```

### 2. Gated Update (LSTM-style)

```python
def gated_update(m_old, m_attended, x_t):
    """
    m_old:      Previous memory
    m_attended: Result of self-attention
    x_t:        Current input
    """
    # Inputs to gates include OLD memory and NEW input
    combined = torch.cat([m_old, x_t_tiled], dim=-1)
    
    f_gate = torch.sigmoid(W_f(combined) + b_f) # Forget
    i_gate = torch.sigmoid(W_i(combined) + b_i) # Input
    
    # Candidate memory usually comes from the attention output
    candidate = torch.tanh(m_attended)
    
    # Final Update
    m_new = f_gate * m_old + i_gate * candidate
    return m_new
```

---

## Common Issues & Fixes

### Loss Is Not decreasing
```python
# Problem: Gradients vanishing or memory washing out
# Fix 1: Check forget gate initialization
nn.init.constant_(self.gate_forget.bias, 1.0) 

# Fix 2: Check LayerNorm placement
# RMC applies LayerNorm *inside* the recurrence, after attention
x = self.norm1(x + self.attention(x))
```

### Model is Slow
```python
# Problem: Attention is O(N^2) where N is memory slots
# Fix: Keep N (slots) small (e.g., 4-8). 
# RMC is efficient precisely because it attends to *memory*, not history.
```

### "RuntimeError: size mismatch"
```python
# Problem: Concatenation axis wrong
# Fix: Ensure you are tiling the input to match the number of slots
input_tiled = input_emb.unsqueeze(1).expand(-1, num_slots, -1)
memory_aug = torch.cat([memory, input_tiled], dim=-1)
```

---

## Comparison: RMC vs. Others

| Feature | LSTM | Transformer | Relational RNN (RMC) |
| :--- | :--- | :--- | :--- |
| **Memory** | Single vector | Entire history | Set of vectors (Slots) |
| **Reasoning** | Implicit (gates) | Explicit (Attention) | Explicit (Attention) |
| **Cost** | O(Seq) | O(Seq^2) | O(Seq x Slots^2) |
| **Best For** | Short/Medium seq | Parallel/Offline | Long/Online Reasoning |

---

*For paper details, see [paper_notes.md](paper_notes.md). For implementation guide, see [README.md](README.md).*
