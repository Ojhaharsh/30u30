# Day 18: Pointer Networks

> Vinyals, Fortunato, Jaitly (2015) — [Original Paper](https://arxiv.org/abs/1506.03134)

**Time:** 4-5 hours  
**Prerequisites:** PyTorch, LSTM intuition, Attention mechanisms  
**Code:** PyTorch (Implementation + Visualization)

---

## What This Paper Is Actually About

Vinyals et al. (2015) solve the "variable output space" problem (Section 1). In traditional Seq2Seq models, you predict tokens from a fixed list (like an English dictionary). But what if you need to output indices that correspond to your input? 

For example, if you are sorting [0.7, 0.2, 0.9], your output isn't a "word" — it's a choice of which input to pick first. The **Pointer Network** uses the attention mechanism to "point" directly at input elements, allowing the output vocabulary to grow and shrink exactly with the input size.

---

## The Core Idea

The paper's core contribution is the repurposing of the attention mechanism (Section 2.3). Instead of using attention to "blend" inputs into a single context vector (which is then used by a fixed softmax layer), the Pointer Network uses the attention distribution itself as the output.

This enables the model to map an input sequence to an output sequence of pointers. Because it points rather than classifies, it can handle any number of inputs at test time, even if that number differs from what was seen during training.

---

## What Authors Actually Showed

The authors demonstrated that Pointer Networks can learn to solve discrete geometric and combinatorial problems simply by observing examples (Section 4):
- **Convex Hull** (Section 4.2): Finding the points that form the boundary of a set of coordinates. Ptr-Net achieved 72.6% accuracy and 99.9% area coverage for n=50 (Table 1).
- **Delaunay Triangulation** (Section 4.3): Computing the triangulation of a point set where no point lies inside any triangle's circumscribed circle. 80.7% accuracy for n=5, dropping to 22.6% for n=10.
- **TSP (Traveling Salesman Problem)** (Section 4.4): Finding an approximate shortest tour through a set of cities. Achieved near-optimal tour lengths for small n (Table 2).

Critically, they showed that a single Ptr-Net model trained on n=5 to 50 generalized to n=500 for Convex Hull with satisfactory results (Table 1), indicating the model learned a reusable algorithm rather than just memorizing input-output pairs.

Note: Our implementation uses sorting as the demonstration task since it's simpler to set up than the geometric problems while still exercising the core pointer mechanism.

---

## The Architecture

### Pointer Attention (Equation 3)
The model treats position-wise attention weights as output probabilities:

$$u_j^i = v^T \tanh(W_1 e_j + W_2 d_i)$$
$$p(C_i | C_1, ..., C_{i-1}, P) = softmax(u^i)$$

- **Encoder**: An LSTM that processes the input sequence (e.g., coordinates).
- **Decoder**: An LSTM that generates a "query" for each output step.
- **Pointer Head**: Computes the additive attention scores between the current query and all encoder states.

### Masking
For permutations (like TSP), once an item is picked, it is "masked" (score set to $-\infty$) so the model doesn't select it again in the same sequence.

---

## Implementation Notes

The implementation in `implementation.py` follows the PyTorch `nn.Module` pattern for the encoder-decoder setup.

Key decisions:
- **Teacher Forcing**: Feeding the ground-truth pointer's embedding during training to speed up convergence.
- **Gradient Clipping**: Standard practice for LSTMs to prevent the "exploding gradient" problem.
- **Masking Mechanism**: Fully implemented to support sampling without replacement.

Things that will bite you:
- **Broadcasting**: Projecting encoder states once ($W_1 e_j$) and then adding the decoder query ($W_2 d_i$) via broadcasting is far more efficient than re-calculating everything at each step.
- **Padding**: Ensure attention masks ignore padding tokens if training on variable-length batches.

---

## What to Build

### Quick Start

```bash
python train_minimal.py --seq_len 5 --epochs 50
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|:---|:---|:---|
| 1 | Pointer Head | Implement the additive attention scoring logic (Eq 3) |
| 2 | Masking | Learn to prevent "sampling with replacement" |
| 3 | Convex Hull | Format geometric data as pointer targets |
| 4 | TSP Cost | Analyze combinatorial results in real-space |
| 5 | Greedy Decoder | Build the full autoregressive selection loop |

---

## Going Further: [Our Implementation: Going Beyond]

While the paper focuses on coordinates and small sets, Ptr-Nets are also useful for:
- **Summarization**: Copying rare words (names, dates) directly from source text.
- **Data Structuring**: Building trees or graphs where nodes "point" to other nodes in the input.

---

## Key Takeaways

1. **Attention as Output**: Attention distribution can replace a fixed-size softmax layer [OK].
2. **Dynamic Vocabulary**: Native support for variable-length inputs and outputs.
3. **Algorithmic Learning**: The model learns to approximate discrete algorithms (like sorting) from raw data.

---

## Files in This Directory

| File | What It Is |
|:---|:---|
| `implementation.py` | Core PointerNetwork and Attention modules |
| `train_minimal.py` | Training script for the sorting task |
| `visualization.py` | Generates attention heatmaps |
| `notebook.ipynb` | Interactive sorting demo |
| `paper_notes.md` | Detailed math and ELI5 |
| `CHEATSHEET.md` | Quick reference and parameter guide |
| `exercises/` | Five exercises covering pointer attention, masking, and decoding |

---

## Further Reading

- [Pointer Networks (Original ArXiv)](https://arxiv.org/abs/1506.03134)
- [Get to the Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

---

**Previous:** [Day 17 — Neural Turing Machines](../17_neural_turing_machines/)  
**Next:** [Day 19: Relational Reasoning](../19_relational_reasoning/)
