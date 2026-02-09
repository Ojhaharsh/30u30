# Day 20: Relational Recurrent Neural Networks

> Adam Santoro et al. (2018) — [Relational recurrent neural networks](https://arxiv.org/abs/1806.01822)

**Time:** 4-6 hours
**Prerequisites:** Day 19 (Relation Networks), basic Attention mechanism
**Code:** PyTorch

---

## What This Paper Is Actually About

Deep learning models struggle with **reasoning over time**. If you tell a model "The red ball is inside the blue box" at Step 1, and then ask "Where is the red ball?" at Step 100, standard RNNs (like LSTMs) often forget the specific relationship ("inside").

This paper introduces the **Relational Memory Core (RMC)**. Instead of treating memory as a single vector (like an LSTM hidden state), it treats memory as a set of **interacting slots**. At every time step, these slots use **Self-Attention** to "talk" to each other, allowing the model to reason about how new information relates to old information *before* deciding what to remember.

It bridges the gap between the storage efficiency of RNNs and the relational power of Transformers.

---

## The Core Idea

The core idea is **Recurrent Self-Attention**.

In a standard Transformer, attention is applied to the entire history of inputs (computationally expensive). In an RMC, attention is applied to a **fixed-size memory matrix** at each step.

1.  **Memory Matrix**: The hidden state is a matrix $M$ (e.g., 4 slots of size 128).
2.  **Interaction**: At each step, the slots attend to each other.
3.  **Gated Update**: The attended information updates the slots using an LSTM-style gating mechanism.

This allows the model to perform complex relational reasoning (like "A relates to B which relates to C") within a fixed memory budget.

---

## What the Authors Actually Showed

The authors demonstrated that RMC outperforms standard LSTMs on tasks requiring long-term relational understanding:
1.  **N-th Farthest**: A synthetic task where the model must remember the N-th vector and compare it to others. RMC achieves **90%+ accuracy**, while LSTMs fail completely as sequence length grows.
2.  **WikiText-103**: On language modeling, RMC matches or beats LSTMs with fewer parameters, showing that structured memory is more efficient.
3.  **Reinforcement Learning**: Solves complex maze tasks (Mini PacMan) better than standard agents.

---

## The Architecture

The **Relational Memory Core (RMC)** consists of three main parts:

1.  **Multi-Head Dot Product Attention (MHDPA)**:
    - Queries ($Q$), Keys ($K$), Values ($V$) are computed from the previous memory $M_{t-1}$ and input $x_t$.
    - The memory slots "attend" to each other and the input.

2.  **Row-wise MLP**:
    - Each memory slot is processed independently by a small feed-forward network.

3.  **Gated Update**:
    - The final memory $M_t$ is computed using input and forget gates (similar to LSTM), ensuring stable training over long sequences.

---

## Implementation Notes

-   **Memory Initialization**: Initializing the memory matrix can be tricky. A common trick is to use a trainable parameter or standard normal distribution.
-   **Attention Implementation**: You can reuse standard PyTorch `MultiheadAttention`, but ensure you apply it correctly over the *memory slots* dimension.
-   **Gating**: The gating is applied per-slot. You can implement this using `Linear` layers applied to the concatenated input/memory.

---

## Gold Standard Update: Baseline & Visualization

We now include a rigorous comparison against a standard **LSTM baseline** to demonstrate the "Relational Gap".

### 1. Model Comparison
The training script (`train_minimal.py`) supports training both RMC and LSTM models side-by-side:
```bash
python papers/20_Relational_RNNs/train_minimal.py --model both --save_plot
```
-   **RMC**: ~14k parameters (Explicit relational memory)
-   **LSTM**: ~22k parameters (Implicit recurrent state)
-   *Goal*: Show that RMC achieves better relational reasoning with fewer parameters on long-context tasks.

### 2. Attention Visualization
To see the RMC's "thought process", you can visualize the self-attention weights:
```bash
python papers/20_Relational_RNNs/train_minimal.py --model rmc --visualize
```
This generates a heatmap (`attention_heatmap.png`) showing how memory slots attend to each other over time.

---

## What to Build

### Quick Start

```bash
# Verify your RMC implementation
python exercises/exercise_01_attention.py
python exercises/exercise_02_gating.py

# Train the full model on N-th Farthest task
python train_minimal.py
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | **Dot Product Attention** | Implement the core self-attention mechanism manually. |
| 2 | **Initialization** | Create the memory matrix and learn why initialization matters. |
| 3 | **Gating Mechanism** | Implement the LSTM-style update equations. |
| 4 | **Build the RMC** | Assemble previous parts into the full `RelationalMemory` module. |
| 5 | **N-th Farthest Data** | Write the generator for the benchmark task. |

Solutions are in `exercises/solutions/`. Try to get stuck first!

---

## Key Takeaways

1.  **Memory as Interaction**: Memory isn't just storage; it's a process of relating new info to old info.
2.  **Attention is All You Need (Inside RNNs)**: You can put attention *inside* a recurrent cell to get better reasoning without the infinite context window of Transformers.
3.  **Structured State**: Breaking state into "slots" helps the model organize information (e.g., Object A in Slot 1, Object B in Slot 2).

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | The complete `RelationalMemory` class. |
| `train_minimal.py` | Training loop for the N-th Farthest task. |
| `paper_notes.md` | Detailed notes, analogies, and math from the paper. |
| `notebook.ipynb` | Interactive walkthrough of the RMC. |
| `exercises/` | Step-by-step implementation tasks. |

---

## Further Reading

- [Original Paper (Santoro et al., 2018)](https://arxiv.org/abs/1806.01822)
- [DeepMind Blog Post](https://deepmind.google/discover/blog/relational-deep-reinforcement-learning/) - A great high-level overview.
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - The origin of the attention mechanism used here.

---

**Previous:** [Day 19: Relational Reasoning](../19_relational_reasoning/)
**Next:** [Day 21: Graph Neural Networks](../21_graph_neural_networks/)
