# Day 19: Relational Reasoning

> Santoro et al. (2017) — [Original Paper](https://arxiv.org/abs/1706.01427)

**Time:** 3-4 hours
**Prerequisites:** PyTorch basics, inductive bias intuition
**Code:** Python + PyTorch

---

## What This Paper Is Actually About

Deep Learning excels at pattern recognition (CNNs) and sequence modeling (RNNs), but it struggles with **relational reasoning**—understanding how entities interact in a structured way. Santoro et al. (2017) argue that standard architectures lack the inductive bias needed to reason about sets of objects.

They introduce the **Relation Network (RN)**, a simple module that explicitly forces a model to consider all possible pairs of objects. By bottlenecking information through pairwise interactions, the model learns to infer relationships (e.g., "left of", "same size as") without requiring massive data or complex symbol processing.

---

## The Core Idea

The RN operates on a set of objects $O = \{o_1, o_2, ..., o_n\}$. It is defined by a composite function:

$$RN(O) = f_{\phi} \left(\sum_{i,j} g_{\theta}(o_i, o_j, q)\right)$$

where:
-   $g_{\theta}$ is an MLP that processes each pair $(o_i, o_j)$.
-   $\sum$ aggregates all pairwise outputs, ensuring the result is independent of order (permutation invariant).
-   $f_{\phi}$ is an MLP that produces the final answer.

---

## What the Authors Actually Showed

In Section 5 (Results), the authors demonstrate that the RN achieves state-of-the-art performance on tasks requiring explicit relational reasoning:

1.  **CLEVR (Visual QA)**: 
    -   **Result**: 95.5% accuracy, surpassing human performance (92.6%).
    -   **Insight**: Standard CNN+MLP baselines failed (68.5%), proving that the pairwise mechanism was the key factor.
2.  **Sort-of-CLEVR**:
    -   **Result**: >94% accuracy on relational questions where baselines plateaued at 63%.
3.  **bAbI (Text QA)**:
    -   **Result**: Solved 18/20 tasks, demonstrating that the same module works for language reasoning.

---

## The Architecture (Section 2)

### 1. Generating Objects
The input to the RN is always a set of objects.
-   **Images**: A CNN processes the image into a $d \times d$ grid of feature vectors. Each cell in the grid is treated as an object.
-   **Language**: Sentences are processed by an LSTM, where hidden states serve as objects.

### 2. Pairwise Function ($g_{\theta}$)
Each object is paired with every other object. The function $g_{\theta}$ (a 4-layer MLP) analyzes the relationship between the pair. If a question is present (e.g., "Is the red sphere left of the blue cube?"), the question embedding $q$ is appended to every pair.

### 3. Aggregation
The outputs of $g_{\theta}$ are summed element-wise. This summation is critical because it makes the model invariant to the order of objects, a necessary property for set reasoning.

---

## Implementation Notes

Our implementation (`implementation.py`) focuses on efficiency and clarity:

-   **Broadcasting**: We avoid Python loops for pair generation. Instead, we use PyTorch broadcasting (`unsqueeze` + `expand`) to generate all $N^2$ pairs in parallel on the GPU.
-   **Coordinate Injection**: As noted in Section 3.1, CNN features lack position info. We manually append $(x, y)$ coordinates to each object vector so the model can learn spatial relations.
-   **Modular Design**: The `RelationNetwork` class is designed to be a plug-and-play module that can be inserted after any feature extractor (CNN or LSTM).

---

## What to Build

### Quick Start

```bash
# Verify environment and permutation invariance
python setup.py

# Train on the "furthest point" relational task
python train_minimal.py --mode furthest --epochs 30
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Pair Generator | Master efficient broadcasting logic for $O(N^2)$ pairing. |
| 2 | Permutation Proof | Programmatically prove the architectural symmetry of the RN. |
| 3 | Sort-of-CLEVR | Implement multi-task question conditioning. |
| 4 | Relational Masking | Learn to handle identity pairs $(o_i, o_i)$ by masking the diagonal. |
| 5 | Counting Logic | Prove that summation preserves cardinality while averaging destroys it. |

Solutions are in `exercises/solutions/`. Try to solve them yourself first.

---

## Key Takeaways

1.  **Inductive Bias Matters**: The RN succeeds not because it is bigger, but because its structure (pairwise bottleneck) matches the problem (relational reasoning).
2.  **Permutation Invariance**: Summation aggregators allow models to process sets without imposing an artificial order.
3.  **Complexity**: The $O(N^2)$ cost is efficient for small sets (like CLEVR objects) but scales poorly to thousands of objects.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Core RN module with type hints and optimized pair generation. |
| `train_minimal.py` | Training script for the "furthest point" and "counting" tasks. |
| `visualization.py` | Tools to visualize relation weights (heatmaps). |
| `setup.py` | Diagnostic script to verify dependencies and invariance. |
| `paper_notes.md` | Detailed notes and mathematical breakdown. |
| `CHEATSHEET.md` | Quick technical reference. |

---

**Next:** [Day 20 - Relational Recurrent Neural Networks](../20_relational_rnn/)
