# Paper Notes: Scaling Laws for Neural Language Models

> **Paper:** Kaplan et al. (OpenAI, 2020)
> **Context:** The breakthrough study that treated LLM training like Physics.

---

## Intuitive Breakdown

Imagine you're building a massive library. 
1.  **N (Model Size)** is how many empty shelves you have.
2.  **D (Dataset Size)** is how many books you buy to fill them.
3.  **C (Compute)** is the number of librarians you hire to sort and index those books.

Previously, people thought that if you had a 10-shelf library and 100 books, you'd be "twice as smart" as a 5-shelf library with 50 books. But it's not that simple. This paper discovery is that if you want to be "twice as smart" (half the error), you might need 10x more shelves and 10x more librarians, but maybe only 2x more books. 

They found the **Physics** behind how quickly the library gets better as you add more of everything. It's not a guess anymore; it's a straight line on a graph.


---

## What the Paper Actually Covers

Jared Kaplan and the OpenAI team (2020) published this work at a turning point in history. Before this, "Architecture Search" was the dominant paradigm — researchers obsessively tweaked transformer components (n_heads, d_ff, etc.). This paper effectively declared those details irrelevant.

**Their core claim:** If you hold other factors constant, the performance of a Language Model follows a smooth, predictable power-law over many orders of magnitude. 

They focused on three primary levers:
- **N**: Number of parameters (excluding embeddings).
- **D**: Number of training tokens.
- **C**: Total training FLOPs.

---

## The Key Discoveries

### 1. The Scaling Power Law
Performance scales as $L \approx X^{-\alpha}$. This relationship holds for $N, D,$ and $C$ individually, provided you aren't bottlenecked by the others.
- **Equation 3.1:** $L(N) = (N_c / N)^{\alpha_N}$
- **Traceability:** $\alpha_N = 0.076$, $\alpha_D = 0.095$.

### 2. The 6N Rule (Equation 2.1)
The industry-standard for compute estimation: $C \approx 6N \times T$.
- **Forward pass:** 2 FLOPs per parameter.
- **Backward pass:** 4 FLOPs per parameter.
Total: 6. This simple rule lets you estimate a \$10M GPU bill on a napkin.

### 3. Sample Efficiency
Larger models are more "sample efficient." They reach the same loss level using fewer training tokens than smaller models (Section 4.1). This suggests that if you have a huge dataset, a huge model will learn from it *faster*.

### 4. Convergence
Models converge in loss but the rate of improvement slows down (vanishing returns). Interestingly, they found that overfitting is only a problem when $N$ is very large relative to $D$.

---

## What the Paper Gets Right vs. Wrong

### What They Got Right
1.  **Predictive Power**: They literally predicted the GPT-3 breakthrough before it happened. The scaling laws they derived from tiny models held true for 175B parameters.
2.  **Architecture Irrelevance**: They correctly identified that width vs. depth matters far less than raw parameter count.
3.  **The "6N" Rule**: Still used daily by nearly every AI researcher in the world.

### What They Got Wrong (The "Chinchilla Gap")
This is the most famous error in modern AI history. Kaplan et al. concluded that when you have 10x more compute, you should scale the model size $N$ by $5.5\times$ and the data $D$ by only $1.8\times$.

**The Reality:** DeepMind later proved (Hoffmann et al., 2022) that this was wrong. Most models in 2020-2021 (including GPT-3) were actually **starved for data**. We now know $N$ and $D$ should scale roughly equally. Kaplan's models were "oversized" and "undertrained."

---

## What the Paper Doesn't Cover

While revolutionary, the Kaplan paper has specific boundaries that are often misunderstood:

1.  **Downstream Capabilities**: The paper only predicts **Cross Entropy Loss**. It does NOT predict when a model will suddenly "learn" to do bench-marking, coding, or reasoning. Those "emergent properties" are not captured by these equations.
2.  **Hardware Efficiency**: It treats every FLOP as equal. In reality, the *latency* and *memory bandwidth* of training a deep vs. wide model on specific hardware (like H100s) varies wildly.
3.  **Data Quality**: It assumes training data is "i.i.d." and of consistent quality. It doesn't explore how cleaning or deduplicating data shifts the scaling curves.
4.  **Optimizer Tuning**: It largely ignores the complex dance of learning rate schedules and batch size scaling, which were later explored in depth by others.

---

## Questions Worth Thinking About

1.  If scaling laws are so predictable, why do we still see "shocks" in capability (like a model suddenly learning to code)?
2.  Since embeddings are excluded from $N$, does a model with a massive vocabulary (10M tokens) follow the same scaling law as one with a small one?
3.  We've scaled $N$ and $D$ for 4 years. Is there a "wall" where the power-law breaks, or can we scale to trillions of parameters infinitely?
4.  If larger models are more sample-efficient, should we always train the largest model possible, even if we can only afford 1 epoch?

---

**Further Reading:** [Chinchilla Scaling Laws (DeepMind)](../26_chinchilla/) - A refinement of the scaling laws for compute-optimality.
