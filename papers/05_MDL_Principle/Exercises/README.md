# Exercises for Day 5: MDL (Minimum Description Length) Principle

Welcome to the MDL exercises! These will help you understand and implement the core concepts.

## Exercise Overview

| # | Exercise | Difficulty | Key Concept |
|---|----------|------------|-------------|
| 1 | Two-Part Code | ⭐ Easy | Basic MDL computation |
| 2 | Prequential MDL | ⭐⭐ Medium | Sequential prediction coding |
| 3 | Model Selection | ⭐⭐ Medium | Polynomial degree selection |
| 4 | NML Complexity | ⭐⭐⭐ Hard | Stochastic complexity |
| 5 | MDL vs AIC vs BIC | ⭐⭐ Medium | Criterion comparison |

## How to Approach

1. **Read the paper_notes.md** first for intuition
2. **Check the CHEATSHEET.md** for formulas
3. **Look at implementation.py** if stuck
4. **Run your solution** to verify

## File Structure

```
Exercises/
├── README.md              (This file)
├── exercise_01_two_part_code.py
├── exercise_02_prequential.py
├── exercise_03_model_selection.py
├── exercise_04_nml_complexity.py
├── exercise_05_mdl_vs_aic_bic.py
└── solutions.py           (Reference solutions)
```

## Tips

- **Don't peek at solutions first!** Try for at least 15 minutes.
- **Use the spy analogy** to build intuition.
- **Remember:** MDL = Model cost + Data cost
- **Test with known data:** Generate polynomial data with known degree.

## Verification

Each exercise has test cases. Run the file to check your solution:

```bash
python exercise_01_two_part_code.py
```

A passing solution will show ✓ for all test cases.

---

Good luck! Remember: **Compression = Understanding** 🗜️
