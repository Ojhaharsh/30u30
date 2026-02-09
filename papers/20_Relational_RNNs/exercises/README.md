# Exercises: Day 20 - Relational RNNs

Temporal reasoning with structured memory.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | **Dot Product Attention** | Easy (2/5) | 20 min |
| 2 | **LSTM Gating** | Medium (3/5) | 30 min |
| 3 | **RMC Step** | Medium (3/5) | 40 min |
| 4 | **N-th Farthest Task** | Easy (2/5) | 20 min |
| 5 | **Training Loop** | Hard (4/5) | 60 min |

---

## Exercise 1: Dot Product Attention

Implement the core mechanism that allows memory slots to talk to each other. You will write the `scaled_dot_product_attention` function manually.

## Exercise 2: LSTM Gating

The RMC uses LSTM-style gating to update its memory. Implement the forget, input, and candidate gates to blend old memory with new attended information.

## Exercise 3: RMC Step

Combine attention and gating to build a single step of the Relational Memory Core.

## Exercise 4: N-th Farthest Task

Create the data generator for the benchmark task. Can you generate sequences and correctly identify the target vector?

## Exercise 5: Training Loop

Put it all together. Train your RMC on the N-th Farthest task and watch the loss go down.

---

## How to Use

1.  Read the exercise file — each has detailed instructions.
2.  Find the TODO sections — these are what you implement.
3.  Run the file — it contains a test function.
4.  Check solutions — compare with `solutions/solution_0X.py`.

## Tips

-   **Shapes**: Watch your tensor shapes! `(batch, slots, heads, dim)` can get confusing.
-   **Gating**: Remember `sigmoid` for gates (0 to 1) and `tanh` for candidate state (-1 to 1).
-   **Initialization**: If loss doesn't decrease, check your initialization.

## Common Issues

-   **Dimension mismatch**: Ensure your query/key/value projections align with `num_heads`.
-   **Exploding Gradients**: The RMC is recurrent. If gradients explode, try clipping them.
