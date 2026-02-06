# Day 15: Bahdanau Attention

> Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014) - [Original Paper](https://arxiv.org/abs/1409.0473)

**Time:** 4-5 hours  
**Prerequisites:** Day 2 (LSTMs/GRUs), Sequence-to-Sequence intuition  
**Code:** PyTorch

---

## What This Paper Is Actually About

Bahdanau et al. (2014) introduce the **attention mechanism** for neural machine translation.

They solve the "bottleneck" problem: previous models tried to compress a whole sentence into one fixed-size vector. This paper demonstrates that translation improves dramatically if the model is allowed to "look back" at specific parts of the input sentence while generating each word of the output.

[Our Retrospective: This paper introduced ideas that influenced later attention-based architectures, including the Transformer (Vaswani et al. 2017).]

---

## What the Authors Actually Showed

Bahdanau et al. evaluated their model (RNNsearch) against a standard encoder-decoder (RNNencdec) on the WMT'14 English-to-French translation task.

The results were decisive:
- **Long Sentences**: While the standard encoder-decoder's performance (BLEU score) plummeted as sentences exceeded 20 words, the attention-based model maintained high accuracy even for sentences of 50+ words.
- **BLEU Scores**: On sentences with no unknown words, RNNsearch-50 achieved **34.16 BLEU**, significantly outperforming RNNencdec-50's **26.71 BLEU**.
- **Alignment Visualization**: They showed that the attention weights correlate with human intuition about word alignment, successfully handling languages with different word orders.

---

## The Core Idea

**Task:** Instead of a single context vector for the whole sentence, compute a different context vector at every single step of the decoder.

```
Input:  [The] [cat] [sat] [on] [the] [mat]
                                    
        [h1]  [h2]  [h3]  [h4]  [h5]  [h6]  (Encoder States)

                     (Alignment / Attention weights)

           Decoder Step i: "chat" -> Focus on [h2] ("cat")
```

To predict well, the model learns an "alignment" function-a small neural network that decides which input words are most relevant to the word currently being translated.

---

## The Architecture

### 1. Bidirectional Encoder (Section 3.2)
The encoder reads the sentence twice - once forward and once backward. This ensures that every encoder hidden state $h_j$ knows the context of the entire sentence, not just what came before it.
- **Forward GRU**: Summarizes context from $x_1$ to $x_j$.
- **Backward GRU**: Summarizes context from $x_T$ to $x_j$.
- **Annotation ($h_j$)**: The concatenation of both directions.

### 2. Additive Attention (Section 3.1)
The model computes an alignment score between the current decoder state and every encoder state using a learned weight matrix:

```
e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
```

**Variables & Dimensions:**
- `s_{i-1}` (hidden): The previous hidden state of the decoder.
- `h_j` (encoder_dim): The $j$-th encoder annotation.
- `W_a`, `U_a`: Weight matrices that project states into the "attention space."
- `v_a`: A weight vector that collapses the attention space into a single scalar score.

### 3. Attention-Based Decoder (Section 3.1)
The decoder is a GRU that receives a unique context vector $c_i$ at every step:
- **Weights ($\alpha_{ij}$)**: Softmax applied over scores $e_{ij}$ (Section 3.1, Eq 6).
- **Context ($c_i$)**: A weighted sum $\sum \alpha_{ij} h_j$.
- **Next State ($s_i$)**: Computed using $s_{i-1}$, the previous output $y_{i-1}$, and the new context $c_i$.

---

## Implementation Notes

When implementing Bahdanau attention, pay attention to these details:

- **The Alignment MLP**: The attention scores are computed by a small Feed-Forward network. This is different from the dot-product attention used in later papers (like Luong or Transformer).
- **Masking**: You MUST mask the attention scores for padding tokens. Set them to `-1e9` or `-inf` before the softmax so the model doesn't waste focus on empty padding.
- **Batch Matrix Multiplication**: Use `torch.bmm` or `@` to compute the weighted sum of encoder states efficiently across the entire batch.
- **Initial Hidden State**: The decoder's initial hidden state $s_0$ is traditionally computed by passing the final backward encoder state through a linear layer and a `tanh` activation (Section 3.2).

**Things that will bite you:**
- **Squeezing/Unsqueezing**: Attention involves many dimension changes (adding/removing the sequence dimension). If you get a "broadcast error," check your `unsqueeze(1)` calls.
- **Softmax Dimension**: Ensure you apply `softmax` over the *source* sequence dimension, not the batch dimension.
- **Vanishing Gradients**: While attention helps, the underlying RNNs can still suffer. Use gradient clipping (threshold of 1.0-5.0).

---

## What to Build

### Quick Start

```bash
# Install dependencies
python setup.py

# Train on sequence reversal task
python train_minimal.py --task reversal --epochs 20
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Implement Additive Attention | Understand the $\tanh$ alignment scoring mechanism |
| 2 | Build Bidirectional Encoder | Handle forward/backward GRU passes and concatenation |
| 3 | Build Attention Decoder | Learn how to integrate the context vector into the GRU step |
| 4 | Sequence Reversal Task | Verify alignment works on a synthetic benchmark |
| 5 | Attention Visualization | Create heatmaps showing what the model focuses on |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1.  **Fixed-length vectors are a bottleneck.** Compressing a sentence to a single vector is fundamentally limited.
2.  **Attention allows dynamic focus.** The decoder "searches" the input for relevant information at each step.
3.  **Alignments emerge automatically.** We don't need to tell the model that "cat" matches "chat"; it learns this by being forced to predict the next word correctly.
4.  **[Our Retrospective] This is a direct ancestor of the Transformer.** The core idea - letting the decoder dynamically attend to encoder states - carried into the Transformer (Vaswani et al. 2017), though the mechanism evolved from additive scoring over RNNs to scaled dot-product self-attention.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Complete Seq2Seq with Bahdanau Attention in PyTorch |
| `train_minimal.py` | Training script with CLI args |
| `setup.py` | Installation and verification script |
| `requirements.txt` | Dependency list |
| `visualization.py` | Attention heatmaps and analysis tools |
| `notebook.ipynb` | Step-by-step walkthrough of the mechanism |
| `exercises/` | 5 implementation exercises |
| `paper_notes.md` | Deep dive into the math and results |
| `CHEATSHEET.md` | Quick ref for equations and dimensions |

---

## Further Reading

- [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) - The original paper
- [Luong et al. (2015)](https://arxiv.org/abs/1508.04025) - Multiplicative attention (a faster variant)
- [Visualizing NMT](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) - Jay Alammar's excellent guide
- [Neural Machine Translation (PyTorch Tutorial)](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

---

**Next:** [Day 16 - Pointer Networks (Order Matters)](../16_order_matters/)
