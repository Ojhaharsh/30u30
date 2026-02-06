# Paper Notes: Neural Machine Translation by Jointly Learning to Align and Translate

## ELI5 (Explain Like I'm 5)

### The Story

Imagine you are translating a long book. 

Traditional models work like this: you read a whole sentence, close your eyes, try to remember everything, and then write the translation in another language. For a short sentence like "The cat sat," this is easy. But for a sentence with 40 words, you will likely forget the beginning by the time you reach the end.

**The Bahdanau Attention approach works differently.** It's like having an "open-book test." For every word you translate, you are allowed to look back at the original sentence and focus your eyes on the specific words that are most relevant to what you are currently writing. 

Note: This analogy is ours, not the authors'. But it captures the main point-that the decoder has a "flashlight" to focus on specific parts of the source sentence.

---

## What the Paper Actually Covers

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014) address the "bottleneck" problem in sequence-to-sequence (seq2seq) models. Previous architectures compressed the entire input sequence into a single fixed-length vector. This paper introduces the **attention mechanism**, which allows the model to search for relevant parts of a source sentence during decoding.

The researchers demonstrate that this approach significantly improves translation quality, especially for long sentences, and provides a way to visualize the "alignments" between languages that the model learns automatically.

---

## The Core Idea (From the Paper)

The fundamental problem with fixed-length vectors is that they cannot capture all the necessary information from a long sequence without loss. The authors propose a "search" process:

1.  **Encode** the source sentence into a sequence of "annotations" (vectors).
2.  **Align** the current decoder state with these annotations to find what's relevant.
3.  **Translate** by generating the next word based on a weighted sum of those annotations.

This process is "joint" because the model learns to translate and align at the same time, without needing pre-existing alignment labels (like those used in older statistical machine translation).

---

## The Math

### 1. The Alignment Model (Score Function)
For each decoder step $i$ and encoder position $j$, we compute an "energy" score $e_{ij}$:
$$e_{ij} = a(s_{i-1}, h_j) = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$
- $s_{i-1}$: The previous decoder hidden state.
- $h_j$: The $j$-th annotation (encoder hidden state).
- $v_a, W_a, U_a$: Weight matrices/vectors learned during training.

### 2. Attention Weights
We normalize these scores using a softmax function to get the attention weights $\alpha_{ij}$:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}$$

### 3. Context Vector
The context vector $c_i$ is a weighted sum of all encoder annotations:
$$c_i = \sum_{j=1}^T \alpha_{ij} h_j$$

### 4. Decoder Hidden State
The new hidden state $s_i$ is computed using the previous state, the previous output $y_{i-1}$, and the context vector:
$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

---

## Key Innovations

### Bidirectional Encoder
The paper uses a bidirectional RNN (BiRNN) to ensure each annotation $h_j$ contains context from both the past and the future of the source sequence.
- **Forward RNN**: Processes $x_1$ to $x_T$ to get $\overrightarrow{h}_j$.
- **Backward RNN**: Processes $x_T$ to $x_1$ to get $\overleftarrow{h}_j$.
- **Annotation**: The concatenation $h_j = [\overrightarrow{h}_j^T ; \overleftarrow{h}_j^T]^T$.

### Dynamic Alignment
Unlike previous models that relied on a single vector for the whole sentence, this model produces a unique context vector $c_i$ for every single word generated. This allows the model to handle word reordering (e.g., French adjectives appearing after nouns) by shifting its attention.

---

---

## The Experiments (English-to-French Translation)

The authors evaluated the attention mechanism on the WMT'14 English-French translation task using a large-scale training set (348M words).

### 1. Setup
- **Baseline (RNNencdec)**: Standard encoder-decoder with a fixed-length vector.
- **Proposed (RNNsearch)**: Encoder-decoder with the new attention mechanism.
- **Lengths**: Evaluated on sentences of up to 30 words and up to 50 words.

### 2. Results (BLEU Scores)
The attention model (RNNsearch) significantly outperformed the baseline, with the gap widening as sentences became longer:

| Model | Sentences < 30 words | All Sentences |
| :--- | :--- | :--- |
| RNNencdec-50 (Baseline) | 26.71 BLEU | 17.82 BLEU |
| RNNsearch-50 (Attention) | 34.16 BLEU | 26.75 BLEU |

Note: These scores are for the filtered test set without unknown words. The key takeaway from Section 4.2 of the paper is that attention effectively solves the performance degradation observed in standard RNN models for long sentences.

---

## Going Beyond the Paper (Our Retrospective)

| Aspect | Bahdanau Attention (2014) | Transformer (2017) |
| :--- | :--- | :--- |
| **Foundation** | Gated RNNs (GRU) | Self-Attention (No RNNs) |
| **Attention Type** | Additive ($v_a^T \tanh(W_s s + W_h h)$) | Scaled Dot-Product ($\frac{QK^T}{\sqrt{d_k}}$) |
| **Information Flow** | Sequential / Recursive | Parallel / Multi-Head |
| **Context** | Fixed-size hidden state + Attention | Fully global attention |
| **Alignment** | Decoder-to-Encoder only | Self-Attention + Cross-Attention |

[Our Retrospective: Bahdanau's work demonstrated that attention significantly improved neural machine translation, particularly for long sentences. It laid conceptual groundwork for the Transformer (Vaswani et al. 2017), which discarded RNNs entirely while keeping attention as its core mechanism.]

---

## Questions Worth Thinking About

1.  **Why bidirectional?** If the encoder only processed the sequence forward, the annotation $h_j$ at the start of the sentence would have zero context from the end of the sentence. Attention is most powerful when keys are context-rich.
2.  **The Memory Bottleneck**: Even with attention, the decoder still uses a fixed-size GRU state. Does this still create a "memory" problem, or does the context vector solve it completely?
3.  **Quadratic Cost**: Bahdanau attention requires $O(N \times M)$ computations (every input token vs every output token). How does this scale for book-length inputs?
4.  **Automatic Alignment**: The model was never given alignment labels (e.g., "cat" matches "chat"). It learned this purely to minimize translation loss. What does this suggest about the "emergence" of internal representations in deep learning?

---

**Next:** [Day 16 - Pointer Networks (Order Matters)](../16_order_matters/)
