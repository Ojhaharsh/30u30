# Day 26: Kolmogorov Complexity | Cheat Sheet | Part of 30u30

Quick reference for Algorithmic Information Theory (AIT) and compression bounds.

---

## The Big Idea (30 seconds)

Kolmogorov Complexity $C(x)$ is the length of the shortest computer program that generates $x$. It formalizes the intuition that **structure is compressible** and **randomness is not**. While uncomputable in its pure form, it provides the target that all data compression and machine learning models aim for.

---

## Key Formulas

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Complexity** | $C(x) = \min \{|p| : U(p) = x\}$ | Length of shortest program |
| **Randomness** | $C(x) \approx |x|$ | No patterns found; incompressible |
| **Invariance** | $C_U(x) \leq C_A(x) + O(1)$ | Choosing a different language only adds a constant |
| **NCD** | $\frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$ | Similarity based on shared patterns |

---

## Huffman vs. Arithmetic Coding

| Type | Best For | Logic |
|------|----------|-------|
| **Huffman** | I.I.D. characters | Tree-based variable length codes |
| **Arithmetic** | Highly skewed data | Range-based (fractional bits) |

---

## Common Issues & Fixes

### NCD is always ~1.0
- **Cause:** Using a compressor that doesn't share global state (like compressing individual strings separately).
- **Fix:** Ensure the compressor sees the concatenated string $xy$ to find mutual patterns.

### Arithmetic Code bit-lengths are high
- **Cause:** Large alphabets or high entropy strings.
- **Fix:** This is actually correct; complex data *requires* more bits.

---

## Debugging Checklist

- [ ] Does `K(Patterned) < K(Random)` for your estimators?
- [ ] Is NCD between identical strings close to 0?
- [ ] Are you handling bit-level precision in your codes?
- [ ] Is the "Invariance" constant accounted for in theoretical comparisons?

---

## Success Criteria

- [OK] Huffman Coder preserves all information (lossless).
- [OK] Random strings achieve $\approx 1.0$ compression ratio (no compression).
- [OK] Highly repetitive strings achieve close to 0 compression ratio.

---

## File Reference

| File | Use It For |
|------|-----------|
| `implementation.py` | Building Huffman/Arithmetic trees from scratch |
| `visualization.py` | Plotting the Complexity Spectrum |
| `train_minimal.py` | Running data diagnostic sweeps |
| `paper_notes.md` | Deep dive into Shen et al. |

---

**Next:** [Day 27 — Machine Super Intelligence](../27_super_intelligence/)
