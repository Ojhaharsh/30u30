# Paper Notes: Relational Recurrent Neural Networks

## ELI5 (Explain Like I'm 5)

### The Memory Cocktail Party

Imagine a standard RNN (like an LSTM) as a person trying to remember a long story. They have a notebook (the hidden state), and at each step, they write down new information and cross out old stuff. But they only look at *one page* at a time. If they need to connect a detail from Page 1 to Page 50, they have to hope they copied that detail forward over and over again for 49 pages.

The **Relational Memory Core (RMC)** is like a cocktail party inside your head. instead of one single notebook page, you have distinct "memory slots"—like different people at a party. At every time step, before taking in new information, these memory slots *talk to each other*.
- Memory A says: "I'm holding the color red."
- Memory B says: "I'm holding the shape square."
- They interact (via attention) and realize: "Oh, we're looking for a red square!"

By letting memories interact *at each step*, the network doesn't just store facts; it reasons about how those facts relate to each other, even if they happened thousands of steps apart.

> **Note:** This analogy is ours, not the authors'.

---

## What the Paper Actually Covers

The paper (Santoro et al., 2018) introduces the **Relational Memory Core (RMC)**. It addresses a specific weakness in standard RNNs: **compartmentalized memory**.

- **Section 1 (Introduction):** Argues that standard memory cells (like LSTM cells) effectively treat memory as a single dense vector (or unorganized slots). Interactive reasoning is hard because the bits of information don't explicitly "attend" to each other during storage.
- **Section 2 (Relational Memory Core):** Proposes the RMC. It replaces the standard LSTM cell update with a multi-head self-attention mechanism.
- **Section 3 (Experiments):** Shows results on:
    - **N-th Farthest**: A synthetic task explicitly designs to break standard LSTMs.
    - **Reinforcement Learning**: Mini PacMan (requiring spatial and temporal reasoning).
    - **Language Modeling**: WikiText-103, where it achieves state-of-the-art (SOTA) results for LSTM-based models.

---

## The Core Idea

The Big Idea is **Self-Attention in the Recurrent Update**.

In a standard Transformer, attention happens over time (looking back at all previous words).
In an RMC, attention happens **over memory slots** at the *current* time step.

1.  **Fixed Slots**: The memory is a matrix $M$ of size $F \times W$ (F rows/slots, W features).
2.  **Interaction**: At each step $t$, the slots run a self-attention mechanism ($M_{t-1}$ attends to itself).
3.  **Update**: The attended memory is then updated using gates (similar to LSTM) to produce $M_t$.

This allows the model to perform "relational reasoning" (connecting concepts) at every single tick of the clock.

---

## The Math

### 1. Multi-Head Dot Product Attention (MHDPA)
Standard Transformer attention, but applied to the memory matrix $M$.

$$ A(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$

In the RMC:
- Input $x_t$ is concatenated to the memory slots (or projected into them).
- Queries ($Q$), Keys ($K$), and Values ($V$) are computed from the memory $M_{t-1}$.
- Output is $\tilde{M}_t$, the "attended" memory.

### 2. The Gating Mechanism (LSTM-style)
The paper uses an LSTM-like update to ensure stability. Instead of scalar gates, we have *row-wise* gates for the memory slots.

For each memory slot $m$:
$$ f_t = \sigma(W_f x_t + U_f m_{t-1} + b_f) $$
$$ i_t = \sigma(W_i x_t + U_i m_{t-1} + b_i) $$
$$ \tilde{m}_t = \text{tanh}(W_c x_t + U_c m_{t-1} + b_c) $$
$$ m_t = f_t \odot m_{t-1} + i_t \odot \tilde{m}_t $$

*(Note: The actual implementation in the paper often uses the attention output as the input to these gates, effectively replacing the linear $x_t$ terms or augmenting them.)*

---

## The Experiments

### N-th Farthest (Section 3.1)
- **Setup**: A sequence of vectors is shown. The model must identify which vector is farthest from the $N$-th vector in the sequence.
- **Why it matters**: To solve this, you *must* remember the N-th vector, hold it, and compare it to every other vector. Standard LSTMs fail because they degrade the "held" vector over time.
- **Result**: RMC solves this with near 100% accuracy, while LSTMs collapse as the sequence length grows.

### WikiText-103 (Section 3.3)
- **Setup**: Word-level language modeling on a large dataset.
- **Result**: RMC achieves perplexity comparable to or better than large LSTMs with fewer parameters, showing that structured memory is efficient.

---

## Going Beyond the Paper (Our Retrospective)

> **Note:** This section is our analysis, looking back from 2024.

The RMC is a fascinating "missing link" between RNNs and Transformers.
- **RNNs**: State is a vector, update is linear/nonlinear.
- **Transformers**: State is the entire history, update is attention.
- **RMC**: State is a matrix (fixed size), update is attention.

It represents the attempt to bring the power of Attention (O(N^2) interactions) into a Recurrent layout (O(1) memory size). While pure Transformers eventually won the "scale" war (because keeping *all* history is better than compressing it, if you have the compute), architectures like RMC are seeing a resurgence in **Linear Attention** and **State Space Models (Mamba/S4)**, which try to compress history into a fixed state efficiently.

---

## Questions Worth Thinking About

1.  **Why fixed slots?** Why not let the number of memory slots grow? (Hint: Computational cost and memory bounds).
2.  **Is it truly relational?** Does attention *guarantee* relational reasoning, or just correlation?
3.  **The "Binding Problem"**: How does the model know that "Slot 1" contains the "Subject" and "Slot 2" contains the "Object"?
