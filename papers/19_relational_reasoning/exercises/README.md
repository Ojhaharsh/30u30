# Day 19 Exercises: Relational Reasoning

Test your understanding of Relation Networks (Santoro et al., 2017).

---

## Exercise 1: Pairwise Broadcasting
**File:** `exercise_01_pair_generator.py`  
**Difficulty:** Medium  
**Goal:** Implement the efficient $N^2$ pair generation logic without using any Python `for` loops. You must use PyTorch's `unsqueeze`, `repeat`, and `cat`.

## Exercise 2: Proving Permutation Invariance
**File:** `exercise_02_permutation.py`  
**Difficulty:** Medium  
**Goal:** Write a script that programmatically proves the RN is order-invariant. If you swap object 1 and object 5 in the input list, the output of the network should remain identical (within numerical precision).

## Exercise 3: Sort-of-CLEVR Logic
**File:** `exercise_03_sort_of_clevr.py`  
**Difficulty:** Hard  
**Goal:** Adapt the forward pass to handle a question embedding. You must concatenate the same question vector to *every* object pair before passing them through $g_{\theta}$.

## Exercise 4: Relational Masking
**File:** `exercise_04_masking.py`  
**Difficulty:** Hard  
**Goal:** Modify the aggregation logic to ignore "self-relations" (where $i=j$). This requires creating a mask that zeros out the diagonal of the $N \times N$ relation grid before summing.

## Exercise 5: Counting vs. Comparing
**File:** `exercise_05_counting.py`  
**Difficulty:** Advanced  
**Goal:** Prove that using `sum` as an aggregator allows the model to "count" objects, whereas `max` or `mean` aggregators struggle with set-size-dependent features.

---

## Instructions

1.  Open the `exercise_XX.py` file.
2.  Complete the `TODO` sections.
3.  Run the file – it contains a test function that will tell you if your implementation is correct.
4.  Compare with `solutions/solution_XX.py`.

## Common Issues
- **Memory Growth**: $N^2$ pairs can explode memory on large sets.
- **In-place Operations**: Avoid modifying tensors in-place during the pair generation phase, as it breaks the calculation of gradients.
- **Aggregation Choice**: If the model can't count, verify you aren't using `mean()` (which averages out counts).
