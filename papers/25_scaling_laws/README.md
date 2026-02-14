# Day 25: Scaling Laws for Neural Language Models

> Kaplan et al. (OpenAI, 2020) — [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

**Time:** 6-8 hours
**Prerequisites:** Transformer Architecture (Day 14), Computational Complexity (Day 24)  
**Context:** The paper that "bet the house" on GPT-3.

---

## What This Paper Is Actually About

Before 2020, training Large Language Models (LLMs) was essentially expensive guesswork. Researchers would stack layers, tweak learning rates, and pray for the loss to go down. Jared Kaplan and the OpenAI team changed this by discovering that LLM performance isn't magic — it's **Physics**.

They proved that if you want to know how a model with 100 Billion parameters will perform, you don't have to build it first. You can train a dozen tiny versions (100k, 1M, 10M params), plot their results on a graph, and **draw a straight line into the future**. 

This discovery is the reason GPT-3 was built. OpenAI knew exactly how smart it would be before they even started the cluster.

---

## The Core Idea: The Power-Law Universe

Language model performance follows a **Power Law** relative to three variables:
1.  **N (Parameters)**: How many "synapses" the model has.
2.  **D (Dataset Size)**: How many tokens it reads.
3.  **C (Compute)**: How many operations (FLOPs) it performs.

### The predictive straight line
On a log-log plot (where $10^1, 10^2, 10^3$ are evenly spaced), Scaling Laws appear as perfectly straight lines. If you double the model size, the loss drops by a constant percentage. Period.

```text
Log(Loss)
  ^
  | *  (Tiny Model)
  |  *
  |   *
  |    *
  |     *  (Medium Model)
  |      *
  |       *
  |        * (175B Parameter GPT-3)
  +------------> Log(Parameters)
```

---

## What This Paper Covers

Kaplan et al. provides a rigorous, empirical "Physics of Language Models":

1.  **Fundamental Scaling Laws**: High-level power laws for parameters ($N$), data ($D$), and compute ($C$).
2.  **Architecture Math**: Empirical proof that $N$ is the bottleneck, not specific transformer configurations (width/depth).
3.  **Optimum Training**: How to allocate a fixed compute budget — should you train a larger model for less time, or a smaller one for more?
4.  **Compute Efficiency**: The first formal definition of the "6N" rule for compute estimation.

---

## What the Authors Actually Showed

Kaplan et al. didn't just theorize; they performed an exhaustive empirical sweep across GPT-style models from 700 parameters to 1.5 Billion. Their findings redefined the frontier:

1.  **Uniformity of Scaling**: They showed that scaling laws are incredibly robust. Whether you change depth, width, or number of heads, the loss depends almost entirely on the total non-embedding parameter count $N$.
2.  **Overfitting Limits**: They discovered that overfitting is remarkably predictable. If you have a model of size $N$, you need roughly $D \approx N^{0.74}$ tokens to avoid bottlenecking (though this was later refined by Chinchilla).
3.  **Low-Compute Prediction**: They demonstrated that you can predict the performance of a model 1,000x larger by simply training a handful of tiny models for a few hours. This is how OpenAI knew GPT-3 would work before they spent a single dollar on its training run.
4.  **Transfer Learning**: They found that the scaling exponents are consistent across different data distributions (e.g., WebText2 vs. Books), but the specific "offsets" ($L_\infty$) vary.

---

## The "6N" Rule: Napkin Math for Billionaires

How do you calculate the cost of a model? Section 2.1 gives us the industry-standard formula:
$$C \approx 6 \times N \times T$$
Where $T$ is the number of tokens. Why **6**?

- **Forward Pass:** 2 ops per parameter (1 multiply + 1 add).
- **Backward Pass:** 4 ops per parameter (twice as much work to compute gradients).
- **Total:** $2 + 4 = 6$ operations per parameter per token.

### The Scaling Architecture Math
Wait, how many parameters are actually in a Transformer? Kaplan et al. use a specific count that excludes embeddings.
```text
For a Transformer with Layer L and Model Dimension d:
+-------------------+----------------+
| Component         | Params         |
+-------------------+----------------+
| Self-Attention    | 4 * d^2        | (Wq, Wk, Wv, Wo)
| Feed-Forward      | 8 * d^2        | (d -> 4d -> d)
| Layer Norms       | ~0             | (Minimal contribution)
+-------------------+----------------+
| Total per Layer   | 12 * d^2       |
+-------------------+----------------+
Total N ≈ 12 * L * d^2
```

---

## The Grit: Why Scale is Hard

Scaling isn't just about drawing lines. It's about surviving the **Bottlenecks**.

1.  **The Dataset Wall**: If you have a huge model ($N$) but only a few books to read ($D$), performance plateaus. Your model memorizes the data (overfitting) and the scaling law breaks.
2.  **The Compute Wall**: Training even a "small" 1B parameter model requires ~0.1 PF-days. That’s more compute than your laptop could do in its entire lifetime.
3.  **The Precision Wall**: If your gradients explode or vanish, those smooth power-law lines become jagged messes.

---

## What You'll Build Today

We move beyond simple "black box" training to create a scaling sweep that mirrors the OpenAI lab environment.

### 1. The KaplanTransformer
A scratch-built architecture where you can verify the **12Ld^2** parameter math down to the bit. We intentionally exclude embeddings to match the paper's rigor.

### 2. The MasterFitter
Predicting the future requires fitting **Irreducible Loss** ($L_\infty$). This is the baseline entropy of language that no model, no matter how large, can beat.

### 3. The Scaling Dashboard
A 4-panel diagnostic suite that visualizes:
- **N-Scaling**: Watch loss drop as you add parameters.
- **Compute Frontier**: See the "bang for your buck" in PF-days.
- **Optimal Allocation**: Predict how much data you *should* have for your model size.

---

## 📁 Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Pure NumPy implementation of $12Ld^2$ counting and power-law fitting |
| `train_minimal.py` | Scaling sweep simulator with bottleneck detection |
| `visualization.py` | 4-panel Scaling Dashboard (The Compute Frontier) |
| `notebook.ipynb` | Interactive Scaling Proof — visualize laws yourself |
| `exercises/` | 5 Tiered exercises to master scaling math |
| `paper_notes.md` | Detailed notes, intuitive breakdown, and the "Chinchilla Gap" retrospective |
| `CHEATSHEET.md` | Quick reference for formulas ($6N, 12Ld^2, \alpha$) |

---

## 📚 Further Reading

1.  **Chinchilla Scaling Laws** (DeepMind): [Coming Tomorrow in Day 26!](../26_chinchilla/)
2.  **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022): The definitive update to Kaplan.
3.  **Scaling Laws for Neural Language Models** (Original Paper): [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
4.  **Scaling Laws for Transfer** (Hernandez et al., 2021): Exploring how scale interacts with transfer learning.
5.  **A Primer on Scaling Laws** (Lilian Weng): [Blog Post](https://lilianweng.github.io/posts/2023-01-27-scaling-laws/)

---

**Next Up:** [Day 26: Chinchilla Scaling Laws](../26_chinchilla/) — *Where we discover OpenAI was actually training their models all wrong...*
