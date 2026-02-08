# Day 19: Relational Reasoning

> *Santoro et al. (2017)* [Original Paper](https://arxiv.org/abs/1706.01427)

**Time:** 3-4 hours | **Prerequisites:** PyTorch basics, inductive bias intuition | **Code:** Python + PyTorch

---

## What This Paper Is Actually About

### The Story: Relationship over Recognition
Imagine looking at a photo of a messy kitchen. A standard AI model (CNN) is like a specialist who is amazing at recognizing individual things: "That's a toaster," "That's a spoon," "That's a tilted cup."

But if you ask, **"Is the spoon closer to the toaster than to the cup?"**, the specialist often fails. Why? Because identifying *what* things are doesn't automatically tell you *how they relate*.

Santoro et al. (2017) argue that **Relational Reasoning** shouldn't be left to chance. Instead, they propose a module that *forces* the network to look at every possible pair of objects. By bottlenecking the information through these pairwise check-ins, the model is forced to learn the logic of relationships—distance, comparison, and counting—achieving superhuman performance on reasoning benchmarks like CLEVR.

---

## The Core Idea

The **Relation Network (RN)** treats the world as a set of objects. It doesn't care about their order; it only cares about their interactions.

### Structural Logic (ASCII Architecture)
```text
Inputs: O = {o1, o2, ..., on} (Objects)

1. BROADCAST: Generate all N^2 pairs
   [o1,o1] [o1,o2] ... [o1,on]
   [o2,o1] [o2,o2] ... [o2,on]
      ...     ...        ...
   [on,o1] [on,o2] ... [on,on]

2. LOCAL RELATION: Apply shared g_theta to each pair
      |       |           |
      v       v           v
    [r1,1]  [r1,2]  ... [rn,n]

3. AGGREGATE: Symmetric Sum (Permutation Invariant)
           \      |      /
            \_____|_____/
                  |
                  v
           [Global Relation]

4. FINAL REASONING: Apply f_phi
                  |
                  v
              Prediction
```


Where:
- $O = \{o_1, o_2, ..., o_n\}$ is the set of objects.
- $q$ is an optional question/context vector.
- $g_{\theta}$ is the **relation function** (an MLP) that processes each pair.
- $f_{\phi}$ is the **global function** (an MLP) that processes the aggregated sum.

$$RN(O) = f_{\phi} \left(\sum_{i,j} g_{\theta}(o_i, o_j, q)\right)$$

---

## What the Authors Actually Showed

In Section 4, the authors demonstrated super-human performance on the CLEVR task, reaching 95.5% accuracy compared to the 68.5% baseline. They also showed:
- **bAbI Generalization**: Solving 18/20 tasks with high accuracy using a single architecture.
- **Physical Reasoning**: Success on the "Sort-of-CLEVR" and "Springs" tasks, proving the model can infer hidden constraints through movement.
- **Inductive Bias**: Proving that the `sum` aggregator (Section 2.1) is essential for cardinality logic, as it preserves the "count" of relations detected.

---

## The Architecture

### Inductive Biases
1. **$O(N^2)$ Interaction**: The architecture explicitly checks every pair ($i, j$).
2. **Permutation Invariance**: Because we sum ($ \sum $), the output is the same regardless of the order of objects.
3. **Weight Sharing**: The same $g_{\theta}$ is applied to every pair.
4. **Coordinate-Awareness**: Appending $(x, y)$ coordinates (Section 3.1) provides spatial grounding for relational tasks on grids.

### The Aggregator Choice
As discussed in Section 2.1, using `sum` instead of `mean` or `max` is a deliberate choice. `sum` preserves cardinality information, allowing the model to distinguish between "3 red cubes" and "1 red cube" even if their feature vectors are identical.

## Implementation Notes

The implementation in `implementation.py` provides a modular RN that can be integrated into vision or text models:

- **Pairwise Broadcasting**: We use PyTorch `unsqueeze` and `expand` to generate $N^2$ pairs without explicit Python loops, ensuring efficiency.
- **Aggregation Strategy**: We implement the `sum` aggregation as described in Section 2.1, which is mathematically the source of the model's permutation invariance.
- **Dropout & Consistency**: We include the 50% dropout in $f_{\phi}$ as specified in Section 4.1 for the CLEVR experiments.

---

## What to Build

### Quick Start
Before running the experiments, run the diagnostic suite to verify permutation invariance.

```bash
python setup.py
```

## Exercises (in [`exercises`](./exercises))

| # | Task | What You'll Get Out of It |
|---|------|-----------|
| 1 | Pair Generator | Master efficient broadcasting logic for O(N^2) pairing. |
| 2 | Permutation Proof | Programmatically prove the architectural symmetry of the RN. |
| 3 | Sort-of-CLEVR | Implement multi-task question conditioning. |
| 4 | Masking Relations | Learn to handle identity pairs (o_i, o_i) in self-relational sets. |
| 5 | Counting logic | Prove that summation preserves cardinality while averaging destroys it. |

---

## Key Takeaways

1. **Relational Reasoning is an architectural constraint.** By forcing the model to process pairs, we give it the logic it needs to solve VQA without needing billions of parameters.
2. **Set-based inputs require order-agnostic aggregators.** The RN uses summation to ensure it doesn't care about the sequence of objects (permutation invariance).
3. **Complexity is $O(N^2)$.** This is the main limitation. As the number of objects increases, the number of pairs grows quadratically.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Comprehensive RN module with type hints and pair-gen logic. |
| `train_minimal.py` | Robust CLI for training and task variant experiments. |
| `visualization.py` | Research suite (Relation heatmaps, distribution plots). |
| `setup.py` | Diagnostic tools (Proof of permutation invariance). |
| `paper_notes.md` | Theoretical walkthrough and mathematical breakdown. |
| `CHEATSHEET.md` | Quick reference for RN dimensions and constraints. |

---

**Previous:** [Day 18 - Pointer Networks](../18_pointer_networks/)  
**Next:** [Day 20 - Relational Recurrent Neural Networks](../20_relational_rnn/)
