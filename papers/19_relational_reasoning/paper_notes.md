# Paper Notes: Relational Reasoning

> Notes on Santoro et al. (2017)

---

## ELI5 (Explain Like I'm 5)

### [Our Addition: The LEGO Analogy]

Imagine you have a box of LEGOs. Most AI models are like someone who looks at the whole box and tries to guess what's inside (CNN) or someone who looks at one block at a time (RNN). 

The **Relation Network** is like someone who picks up *every possible pair* of blocks, looks at how they connect, and writes a note about that pair. After looking at every pair, they sum up all their notes to understand the whole structure. This "pairwise" thinking is what allows the model to understand things like "which block is furthest from the red one?" or "are there two blocks of the same size?"

> **Note:** This analogy is ours.

---

## What the Paper Actually Covers

Santoro et al. (2017) address a fundamental limitation in deep learning: the difficulty of relational reasoning. While standard architectures thrive on structured data (grids for CNNs, sequences for RNNs), they lack a strong inductive bias for reasoning about the relationships between entities in an unstructured set.

The paper introduces the **Relation Network (RN)**, a neural module constrained to pairwise interactions. They test this module across three distinct domains:
1.  **Visual QA (CLEVR)**: Reasoning about spatial relationships, colors, and shapes.
2.  **Text-based QA (bAbI)**: Deductive reasoning across sentences.
3.  **Physical Reasoning**: Predicting the movement of balls connected by invisible springs/constraints.

---

## The Core Idea

The RN is a composite function designed to be plug-and-play. It takes a set of objects $O = \{o_1, o_2, ..., o_n\}$ and computes:

$$RN(O) = f_{\phi} \left( \sum_{i,j} g_{\theta}(o_i, o_j, q) \right)$$

### 1. Object Definition
The "objects" vary by task (Section 2):
-   **CNN Features**: For images, each cell in a $k \times k$ feature map is an object.
-   **NLP**: Each LSTM state or word embedding is an object.

### 2. The Relation Function ($g_{\theta}$)
$g_{\theta}$ is an MLP shared across all pairs $(i, j)$. Its role is to infer whether a relation exists between two objects and what that relation is.

### 3. The Global Function ($f_{\phi}$)
After summing the relation outputs, $f_{\phi}$ (another MLP) aggregates the information to produce a final answer.

### [Important Detail] Spatial Awareness
For visual tasks like CLEVR, the authors found that objects extracted from CNN feature maps lack position information. To fix this, they append absolute $(x, y)$ coordinates to each object's feature vector (Section 3.1).

---

## The Experiments

### 1. Visual QA: CLEVR (Section 5.1)
The authors trained the RN on the CLEVR dataset, which requires multi-step reasoning (e.g., "What shape is the object that is left of the green sphere?").

-   **Result**: The RN achieved **95.5% accuracy**, surpassing human performance (92.6%) and the previous best model (68.5%).
-   **Table 1 Evidence**: The RN significantly outperformed standard stacked attention models (SA) and CNN+MLP baselines.

### 2. Text-based QA: bAbI (Section 5.4)
The RN was applied to the bAbI text reasoning suite.
-   **Result**: It solved **18/20 tasks**.
-   **Failure Cases**: It struggled with "Two supporting facts" and "Three supporting facts", suggesting limitations in multi-hop chaining compared to memory networks.

### 3. Physical Reasoning (Section 5.5)
The model observed balls moving on a table, some connected by invisible springs.
-   **Goal**: Predict future movement or infer connections.
-   **Result**: The RN successfully inferred the hidden physical graph structure, predicting connections with **93% accuracy** in the test set.

---

## Generalization (Section 4 & 5)

A key claim is that RNs generalize to different numbers of objects. In the Supplementary Material (Section F), the authors show that a model trained on sets of 6 objects could generalize to sets of 12 objects, thanks to the permutation-invariant summation aggregator.

---

## What the Paper Doesn't Cover

-   **Computational Cost**: The $O(N^2)$ complexity is mentioned but not deeply analyzed as a bottleneck for large sets (e.g., thousands of objects).
-   **Hierarchical Reasoning**: The RN is "flat"—it considers all pairs equally. It does not explicitly model hierarchies of objects.

---

## [Our Addition: Retrospective]

| Feature | Original RN (2017) | Modern Transformers (Visual/NLP) |
|:---|:---|:---|
| **Mechanism** | Additive Pairwise ($ \sum g_{\theta} $) | Multi-Head Self-Attention |
| **Logic** | Fixed Pairwise bottleneck | Dynamic Weighting (Attention) |
| **Complexity** | $O(N^2)$ | $O(N^2)$ |
| **Inductive Bias** | Explicitly Pairs | Learned structure via Attention |

While the RN was a breakthrough in demonstrating the value of relational bottlenecks, it is effectively a precursor to the Transformer's self-attention mechanism, which also operates on sets of pairs but with data-dependent weighting (attention scores) rather than a fixed sum.

---

## Questions Worth Thinking About

1.  Why does the `sum` operation allow the model to count objects, whereas `mean` or `max` might lose this information?
2.  The paper uses a full all-to-all comparison ($N^2$). Can you think of ways to prune this to $O(N)$ or $O(N \log N)$? (Hint: Sparse graphs or nearest neighbors).
3.  Why is coordinate injection $((x, y)$ features) so critical for CNN-based objects but not always for graph-based inputs?

---

**Next:** [Day 20 - Relational Recurrent Neural Networks](../20_relational_rnn/)
