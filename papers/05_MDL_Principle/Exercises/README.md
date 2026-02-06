# Exercises: MDL Principle

5 exercises implementing concepts from Grünwald (2004). Each exercise maps directly to a section of the paper.

---

## Exercise Overview

| # | Exercise | Difficulty | Paper Section | Source |
|---|----------|------------|--------------|--------|
| 1 | Two-Part Code | Easy | Section 3 (crude MDL) | Paper concept, our implementation |
| 2 | Prequential MDL | Medium | Section 5 (predictive coding) | Paper concept, our implementation |
| 3 | Model Selection | Medium | Running example (polynomial) | Paper's own example |
| 4 | NML Complexity | Hard | Section 6 (stochastic complexity) | Paper concept, our implementation |
| 5 | MDL vs AIC vs BIC | Medium | Section 8 (comparison) | Paper's comparison, our Monte Carlo extension |

---

## How to Work Through These

1. Read `paper_notes.md` for the theory
2. Check `CHEATSHEET.md` for formulas
3. Each exercise file has TODO markers — fill them in
4. Run the file to check against built-in tests

```bash
python exercise_01_two_part_code.py
```

Passing tests show `[ok] PASSED`. Look at `solutions.py` only after attempting each exercise.

---

## Exercise 1: Two-Part Code [Easy]

**Paper Section 3.** Implement the basic MDL score: L(H) + L(D|H).

**Tasks:**
- Compute model description length (parameters x bits)
- Compute data description length (Gaussian residual code)
- Select the polynomial degree with minimum total score

---

## Exercise 2: Prequential MDL [Medium]

**Paper Section 5.** Implement sequential prediction coding.

**Tasks:**
- Predict each data point using only previous points
- Accumulate log-loss (cumulative surprise)
- Compare to two-part MDL scores

---

## Exercise 3: Model Selection [Medium]

**Paper's running example.** Full polynomial degree selection with MDL.

**Tasks:**
- Generate noisy polynomial data (known true degree)
- Compute MDL scores for degrees 1 through 10
- Verify MDL selects the true degree
- Visualize the model-vs-data cost tradeoff

---

## Exercise 4: NML Complexity [Hard]

**Paper Section 6.** Compute stochastic complexity.

**Tasks:**
- Implement approximate NML complexity for Gaussian models
- Compare NML-based selection with two-part code selection
- Observe how stochastic complexity scales with sample size and model dimension

---

## Exercise 5: MDL vs AIC vs BIC [Medium]

**Paper Section 8.** Head-to-head comparison.

**Tasks:**
- Implement AIC, BIC, and MDL scoring
- Run polynomial selection with all three criteria
- Monte Carlo: compare accuracy across many random datasets (our extension)
- Analyze when the criteria agree and disagree

---

Solutions in `solutions.py`.
