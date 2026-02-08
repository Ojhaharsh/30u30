# Day 19 Exercises: Relational Reasoning

Test your understanding of Relation Networks (Santoro et al., 2017).

---

## Exercise 1: Pairwise Broadcasting
**File:** `exercise_01_pair_generator.py`  
**Difficulty:** Medium  
**Goal:** Implement efficient $N^2$ pair generation without loops.

## Exercise 2: The Relation Function (g_theta)
**Files:** `exercise_02_g_theta.py`  
**Difficulty:** Easy  
**Goal:** Implement the MLP that processes each pair.

## Exercise 3: Permutation Invariance
**File:** `exercise_03_permutation.py`  
**Difficulty:** Medium  
**Goal:** Prove that shuffling inputs doesn't change the output.

## Exercise 4: Contextual Questions (Sort-of-CLEVR)
**File:** `exercise_04_sort_of_clevr.py`  
**Difficulty:** Hard  
**Goal:** Inject a question vector into every pair to condition the reasoning.

## Exercise 5: Coordinate Injection
**File:** `exercise_05_coordinates.py`  
**Difficulty:** Easy  
**Goal:** Append (x, y) coordinates to object features.

## Exercise 6: Relational Masking
**File:** `exercise_06_masking.py`  
**Difficulty:** Hard  
**Goal:** Mask out self-relations (diagonal) before aggregation.

## Exercise 7: Counting Bias
**File:** `exercise_07_counting.py`  
**Difficulty:** Advanced  
**Goal:** Prove that `sum` aggregation allows counting, while `mean` does not.



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
