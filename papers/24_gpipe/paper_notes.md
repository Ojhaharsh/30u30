# Paper Notes: GPipe: Efficient Training of Giant Neural Networks

> Huang et al. (2018/2019)

---

## Post Overview

**Title:** GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
**Authors:** Yanping Huang, Youlong Cheng, et al. (Google Brain)
**Year:** 2018 (arXiv), 2019 (NeurIPS)
**Venue:** NeurIPS 2019
**Core Question:** *"How can we train models with billions of parameters that exceed the memory of a single accelerator without sacrificing efficiency?"*

---

## ELI5 (Explain Like I'm 5)

### The Pizza Oven Analogy

Imagine you are making 100 pizzas. You have 4 workers:
- **Worker A** rolls the dough.
- **Worker B** adds the sauce and cheese.
- **Worker C** adds the toppings.
- **Worker D** puts them in the oven.

If you make one pizza at a time, Workers B, C, and D are standing idle while Worker A rolls the dough. This is slow **naive model parallelism**.

**GPipe** says: As soon as Worker A finishes rolling the *first* pizza dough and passes it to Worker B, they immediately start rolling the *second* dough. By the time the first pizza is in the oven (Worker D), all four workers are working on four different pizzas at the same time.

> **Note:** This analogy is ours, not the authors'. [Our Addition: Analogy]

---

## What the Paper Actually Covers

GPipe addresses the scalability limit of deep learning. It focuses on **pipeline parallelism**, which partitions the model layers across multiple devices and uses micro-batching to overlap computation. Key innovations include the **Synchronous Pipeline Schedule** and the integration of **Re-materialization** (activation checkpointing) at the pipeline level to optimize memory.

---

## The Core Idea (Section 3.1)

### Batch Splitting
A mini-batch of size $N$ is split into $M$ micro-batches. These micro-batches are processed through $K$ physical partitions.

### The Pipeline Bubble
Because the first stage must finish before the second starts, there is a delay (the "bubble"). The paper formalizes this overhead:
- **Total Time** for forward pass: $T_{forward} = (M + K - 1) \cdot t_{f}$
- **Bubble Rate**: $(K-1) / (M + K - 1)$

As $M$ becomes much larger than $K$, the bubble rate approaches zero.

---

## The Math

### Memory Reduction (Section 3.2)
Without GPipe, peak activation memory is $O(N \cdot L)$ where $L$ is total layers.
With GPipe and re-materialization:
- Each partition only stores input activations for its $M$ micro-batches.
- Internal activations for the stage are recomputed during the backward pass.
- **Peak Memory**: $O(N \cdot L/K + N/M \cdot L)$

### Theoretical Throughput
The paper shows that throughput scales linearly with $K$ as long as $M$ is sufficiently large.

---

## The Experiments (Section 4)

### 1. AmoebaNet (Table 2)
- **Model**: AmoebaNet-B with 557M parameters.
- **Hardware**: 8x Cloud TPU v2.
- **Result**: Achieved 84.4% Top-1 ImageNet accuracy.
- **Observation**: GPipe allowed a 3.5x increase in model size on the same hardware compared to data parallelism alone.

### 2. Giant Transformers (Table 3)
- **Model**: 6-Billion Parameter Transformer (128 layers).
- **Result**: Significant improvement in BLEU scores for multilingual translation tasks.
- **Bottleneck**: The authors note that communication between TPU cores (via ICI) is high-speed, which helps maintain efficiency.

---

## What the Paper Gets Right
- **Mathematical Identity**: By being synchronous, GPipe doesn't suffer from the convergence issues of asynchronous pipelines.
- **Library Design**: It abstracts the complexity away from the user, making it a "drop-in" scaling solution.

## What the Paper Doesn't Cover
- **Dynamic Partitioning**: The paper assumes we can manually split layers. It doesn't solve the problem of automatically balancing load across GPUs with different speeds.
- **Asynchronous Gains**: Asynchronous models (like PipeDream) can have zero bubbles, but at the cost of "stale" gradients. GPipe ignores this trade-off in favor of simplicity.

---

## Looking Back (Our Retrospective, Not in the Paper)

Since 2019, GPipe has become the "gold standard" foundation. Modern frameworks like **DeepSpeed** and **Megatron-LM** use GPipe-style pipelining as one of the three pillars of 3D Parallelism (Data + Pipeline + Tensor). The "micro-batching" concept is now omnipresent in LLM training.

---

## Questions Worth Thinking About

1. How would you handle **Batch Normalization** if your micro-batch size is only 1?
2. If communication between devices is very slow (e.g., standard Ethernet), does the GPipe benefit still hold?
3. Could you combine GPipe with **Gradient Accumulation** across time steps?

---

**Next:** [Day 25 - Scaling Laws for Neural Language Models](../25_scaling_laws/)
