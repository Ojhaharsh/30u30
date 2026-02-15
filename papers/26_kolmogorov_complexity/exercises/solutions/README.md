# Exercise Solutions: Kolmogorov Complexity and Algorithmic Information

Complete, well-commented solutions for all 5 exercises.

---

## How to Use These Solutions

### Try First, Check Later!

These solutions are here to help you **learn**, not to copy-paste. Here's the recommended approach:

1. **Attempt the exercise first** (spend at least 30-60 minutes).
2. **Get stuck?** Review the relevant section in the main `README.md`.
3. **Still stuck?** Look at just the function you need help with in the solution.
4. **Compare your solution** with ours after completing the exercise.

**Remember:** Mastering Algorithmic Information Theory requires building the intuition for *why* things compress.

---

## Solutions Index

### Exercise 1: Huffman Tree Construction
**File:** `solution_01_huffman.py`  
**Difficulty:** Medium  
**What's included:**
- Complete greedy merging logic using `heapq`.
- Recursive prefix code generation.
- Detailed comments on why least-frequent nodes are merged first.

**Key learning points:**
- How frequency distributions dictate bit-depth.
- The property of prefix codes (no code is a prefix of another).

---

### Exercise 2: Bit-rate vs Shannon Entropy
**File:** `solution_02_entropy.py`  
**Difficulty:** Easy  
**What's included:**
- Shannon Entropy implementation ($H = -\sum p_i \log_2 p_i$).
- Bit-per-character (BPC) calculation.
- Comparison logic explaining the "redundancy" gap.

**Key learning points:**
- Shannon Entropy as the theoretical lower bound for $C(x)$ under i.i.d. assumptions.
- Why Huffman coding gets close to but often stays slightly above the Shannon limit.

---

### Exercise 3: Similarity Check (NCD)
**File:** `solution_03_ncd.py`  
**Difficulty:** Medium  
**What's included:**
- Normalized Compression Distance (NCD) formula implementation.
- Comparative tests across different text domains.
- Breakdown of how cross-compression identifies shared Kolmogorov complexity.

**Key learning points:**
- $C(xy)$ is smaller than $C(x) + C(y)$ if $x$ and $y$ share "mutual information".

---

### Exercise 4: Arithmetic Range Narrowing
**File:** `solution_04_arithmetic.py`  
**Difficulty:** Hard  
**What's included:**
- Iterative interval narrowing logic.
- Log-based theoretical bit-length calculation ($Bits = -\log_2(RangeWidth)$).
- Visual prints of the intervals as they shrink.

**Key learning points:**
- How Arithmetic coding represents sequences as single numbers.
- The precision requirements that make Arithmetic coding superior to Huffman.

---

### Exercise 5: The Incompressibility Challenge
**File:** `solution_05_incompressibility.py`  
**Difficulty:** Hard (Logic)
**What's included:**
- Strategy for generating "random" strings (maximum entropy).
- Verification of the incompressibility ratio ($> 70\%$).

**Key learning points:**
- Understanding that "randomness" is just the absence of a shorter description.
- Most strings of a given length are, by definition, incompressible.

---

## Need Help?

1. **Check the code comments** - they explain the mathematical rationale.
2. **Review the main README** - it contains the pseudocode and architecture.
3. **Run the visualization** - `visualization.py` helps build intuition for these metrics.

---

| Exercise | Solution Available | Verified | Study Time |
|----------|-------------------|----------|------------|
| 1. Huffman Tree | Available | Yes | 1 hour |
| 2. Entropy vs BPC | Available | Yes | 30 mins |
| 3. Similarity (NCD) | Available | Yes | 1 hour |
| 4. Arithmetic Range | Available | Yes | 1-2 hours |
| 5. Incompressibility | Available | Yes | 30 mins |

**Total:** All 5 solutions are complete and formatted to Gold Standards.
