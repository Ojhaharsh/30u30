# Exercises: Kolmogorov Complexity and Algorithmic Information

5 hands-on exercises to master the fundamentals of compression-based complexity. Work through them in order - each builds on the previous!

---

## Exercise 1: Huffman Tree Construction
**Difficulty**: Medium
**Time**: 1 hour
**File**: `exercise_01_huffman.py`

### Goal
Implement a complete Huffman encoder from scratch in NumPy, including tree building and prefix code generation.

### What You'll Learn
- How frequency-based encoding maps to $C(x)$
- The mechanics of prefix codes
- How to use priority queues for optimal tree construction

### Tasks
1. Initialize leaf nodes for each character frequency.
2. Implement the tree-merging logic: always merge the two lowest-frequency nodes.
3. Recursively traverse the tree to assign '0's and '1's.

### Success Criteria
- Encoded string is significantly shorter than raw 8-bit characters.
- Decoding the bit-string would return the original text (lossless property).

### Hints
- Use `heapq` for the priority queue.
- Frequent characters should have shorter bit-paths.

---

## Exercise 2: Bit-rate vs Shannon Entropy
**Difficulty**: Easy
**Time**: 30 mins
**File**: `exercise_02_entropy.py`

### Goal
Compare empirical compression against the theoretical Shannon Limit.

### What You'll Learn
- The relationship between $C(x)$ and $H(X)$.
- Why Huffman is "Shannon-optimal" but not "perfectly optimal".

### Tasks
1. Calculate bits-per-character (BPC) for our Huffman implementation.
2. Implement the Shannon Entropy formula in bits.
3. Verify that $BPC \geq H(X)$.

---

## Exercise 3: Similarity Check (NCD)
**Difficulty**: Medium
**Time**: 1 hour
**File**: `exercise_03_ncd.py`

### Goal
Use Normalized Compression Distance to identify patterns in text.

### What You'll Learn
- How shared patterns reduce joint complexity $C(xy)$.
- Why NCD is a universal similarity metric.

### Tasks
1. Calculate NCD for identical, related, and random strings.
2. Verify that `NCD(Identity) < NCD(Different) < NCD(Random)`.

---

## Exercise 4: Arithmetic Range Narrowing
**Difficulty**: Hard
**Time**: 1-2 hours
**File**: `exercise_04_arithmetic.py`

### Goal
Implement the range update logic for Arithmetic encoding.

### What You'll Learn
- How to encode sequences into fractional number ranges.
- Why Arithmetic coding outperforms Huffman by using fractional bits.

### Tasks
1. Implement the `low` and `high` range updates for each character.
2. Calculate the theoretical bits required to represent the final range.

---

## Exercise 5: The Incompressibility Challenge
**Difficulty**: Hard
**Time**: 30 mins
**File**: `exercise_05_incompressibility.py`

### Goal
Generate a string that is "Kolmogorov Random" relative to our estimators.

### What You'll Learn
- The definition of randomness in AIT.
- Why most strings are incompressible.

### Tasks
1. Experiment with different string generation strategies.
2. Generate a 100-character string that achieves $> 70\%$ bit-ratio.

---

## Solutions
Complete solutions are in the `solutions/` folder:
- `exercise_01_solution.py`
- `exercise_02_solution.py`
- `exercise_03_solution.py`
- `exercise_04_solution.py`
- `exercise_05_solution.py`

**Recommendation**: Try solving on your own first!
