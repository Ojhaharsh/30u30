# Day 17: Neural Turing Machines (NTM)

> Alex Graves, Greg Wayne, Ivo Danihelka (2014) - [Original Paper](https://arxiv.org/abs/1410.5401)

**Time:** 3-5 hours  
**Prerequisites:** PyTorch, LSTM intuition, Linear Algebra (matrix products)  
**Code:** PyTorch (Implementation + Visualization)

---

## What This Paper Is Actually About

The Neural Turing Machine (NTM) addresses a fundamental limitation of recurrent neural networks (RNNs): the coupling of processing power and memory capacity. 

In a standard LSTM, the "memory" is the hidden state vector. If you want to store more information, you have to increase the size of that vector, which also increases the number of weights in the network (the "thinking" part). The NTM breaks this coupling by connecting a neural network **Controller** to a separate **Memory Bank**. 

This allows the network to store large amounts of data without becoming computationally unmanageable. It successfully learns *algorithms* (like sorting and copying) rather than just statistical mappings.

---

## The Core Idea: Differentiable Addressing

The central problem Graves et al. address is: **How do you make a pointer differentiable?**

In a standard computer, a pointer is a discrete address (e.g., "0x7ffd"). You can't take a derivative of a discrete address. To solve this, Graves et al. introduced **Weightings**. 

Instead of reading from one address, the NTM "reads" from every address at once, using a weighting vector $w_t$ that sums to 1. If the weight is 1.0 at address $A$ and 0.0 everywhere else, it's a discrete pointer. If it's spread out, it's a "blurry" focus.

### The Addressing Pipeline (Section 3.3.1)

To determine where to focus, the NTM uses a four-stage process:

1.  **Content Addressing (Eq 5):** The model emits a "search key" $k_t$. It looks for rows in memory that are similar to $k_t$ using cosine similarity.
2.  **Interpolation (Eq 7):** A gate $g_t$ decides whether to use the new search result or stick with the previous focus $w_{t-1}$.
3.  **Convolutional Shift (Eq 8):** A shift weighting $s_t$ performs circular convolution, allowing the "pointer" to move (e.g., "stay put", "move +1", "move -1").
4.  **Sharpening (Eq 9):** A factor $\gamma_t$ "squeezes" the distribution to make the focus more precise.

---

## What the NTM Actually Showed

### 1. The Copy Task (Section 4.1)
The model is shown a sequence of random bits, a delimiter, and is then asked to repeat the sequence. 

- **LSTM Result:** Struggles once the sequence length exceeds the size of its hidden state.
- **NTM Result:** Learns to increment its memory pointer as it writes, and then increment it as it reads. It generalizes to sequences much longer than those it was trained on because it learned the *algorithm* of copying, not just the data.

### 2. The Sorting Task (Section 4.5)
The model is shown a list of vectors, each with a priority. It is then asked to output them in sorted order.

- **Outcome:** The NTM learns to use its memory to store## The Architecture

### 1. The Addressing Pipeline (Section 3.3.1)
The NTM uses a four-stage process to decide where to focus its "blurry" pointer:

1. **Content Addressing**: Emits a "search key" $k_t$. Looks for rows in memory similar to $k_t$ using cosine similarity (Eq 5).
2. **Interpolation**: A gate $g_t$ decides whether to use the new search result or stick with the previous focus (Eq 7).
3. **Convolutional Shift**: A shift weighting $s_t$ performs circular convolution, allowing the pointer to move (e.g., stay put, +1, -1) (Eq 8).
4. **Sharpening**: A factor $\gamma_t$ "squeezes" the distribution to make the focus more precise (Eq 9).

### 2. Controller & Heads
- **Memory Matrix ($N \times M$):** The "scratchpad" where data is stored.
- **Read Head**: Computes a weighted average of memory: $r_t = \sum w_t(i) M_t(i)$ (Section 3.1).
- **Write Head**: Performs an **Erase** followed by an **Add** (Section 3.2):
  - `M_t(i) = M_{t-1}(i) * [1 - w_t(i) * e_t]` (Erase)
  - `M_t(i) = M_t(i) + w_t(i) * a_t` (Add)

**Variables & Dimensions:**
- `w_t` (N): The weighting vector (the pointer).
- `e_t` (M): The erase vector (0 to 1).
- `a_t` (M): The add vector (data to write).
- `k_t` (M): The content search key.

---

## Implementation Notes

When implementing an NTM, pay attention to these stability details:

- **Gradient Clipping (Section 4)**: NTMs are notoriously hard to train. A clip of 10.0 is almost mandatory to prevent divergence during the circular shift operation.
- **Numerical Stability**: Operations like sharpening ($w^\gamma$) can generate extremely large or small numbers. Use `torch.clamp` and small epsilons ($1e-8$) to prevent `NaN`.
- **Initialization**: Initial memory and weights should be learned parameters (`nn.Parameter`), not just zeros.

**Things that will bite you:**
- **Circular Convolution**: Implementing Eq 8 correctly requires a specialized 1D convolution or a sequence of `torch.roll` operations.
- **Softplus for Positivity**: Parameters like $\beta$ (key strength) and $\gamma$ (sharpening) must be positive. Use `1 + F.softplus(x)` to ensure stability.
- **Memory Footprint**: A $128 \times 20$ memory bank is small, but the addressing logic (calculating scores for every slot) adds significant overhead per step.
ry and weights are learned. In our implementation, we use learned buffers for `init_memory` and `init_w` to match this.

---

## What to Build

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the copy task training
python train.py
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|---------------------------|
| 1 | Content-based addressing | Implement cosine similarity weighting (Eq 5) |
| 2 | Convolutional shift | Build circular convolution for location-based addressing (Eq 8) |
| 3 | Memory erase and add | Implement the write head update operations (Eq 3, 4) |
| 4 | Controller input preparation | Concatenate input with previous read vectors (Section 2) |
| 5 | Sharpening mechanism | Apply gamma exponentiation to prevent blurry focus (Eq 9) |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **Decoupling memory from computation is the core contribution.** Standard RNNs store everything in the hidden state, coupling memory capacity with processing power. The NTM separates these by connecting a controller to an external memory bank (Section 1).

2. **Differentiable addressing makes discrete pointer operations learnable.** By using soft weightings that sum to 1, the NTM can read and write to memory locations while remaining fully differentiable for backpropagation (Section 3.1).

3. **The four-stage addressing pipeline is necessary, not decorative.** Content addressing alone cannot implement simple algorithms like incrementing a pointer. Interpolation, shift, and sharpening together allow learned sequential access patterns (Section 3.3.1).

4. **NTMs learn algorithms, not just statistical mappings.** On the copy task, the model generalizes to sequences longer than those in training because it learns to increment a pointer, not memorize sequence patterns (Section 4.1).

---

## Files in This Directory

| File | What It Is |
|------|------------|
| `implementation.py` | Full NTM with controller, memory, and read/write heads |
| `train.py` | Training script for the copy task |
| `visualization.py` | Memory state and addressing weight visualizations |
| `notebook.ipynb` | Interactive walkthrough of addressing mechanisms |
| `exercises/` | 5 exercises with solutions |
| `paper_notes.md` | Condensed notes on the original paper |
| `CHEATSHEET.md` | Quick reference for architecture dimensions and training tips |
| `requirements.txt` | Python dependencies |

---

## Further Reading

- [Original Paper - Neural Turing Machines (arXiv:1410.5401)](https://arxiv.org/abs/1410.5401)
- [Differentiable Neural Computers (Graves et al., 2016)](https://www.nature.com/articles/nature20101) - the successor to NTMs with improved memory addressing
- [Learning to Transduce with Unbounded Memory (Grefenstette et al., 2015)](https://arxiv.org/abs/1506.02516) - alternative neural stack/queue architectures

---

**Next:** [Day 18 - Pointer Networks](../18_pointer_networks/)
