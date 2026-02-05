# Exercise Solutions

Complete, well-commented solutions for all 5 exercises.

---

## How to Use These Solutions

**Try first, then look.** The best way to learn:

1. Attempt the exercise yourself (spend at least 20-30 minutes)
2. Get stuck? Review the relevant section in [../README.md](../README.md)
3. Still stuck? Look at just the function you need help with
4. Compare your solution with ours after completing the exercise

If you solved it differently, that's fine — there are many valid approaches. Run both to compare.

---

## Solutions Index

| File | Exercise | Difficulty | Study Time |
|------|----------|-----------|------------|
| `01_dropout.py` | Dropout Layer | Medium | 20-30 min |
| `02_normalization.py` | Layer Normalization | Hard | 30-45 min |
| `03_weight_decay.py` | Weight Decay (L2) | Easy | 15-20 min |
| `04_early_stopping.py` | Early Stopping | Medium | 25-35 min |
| `05_full_pipeline.py` | Full Pipeline | Very Hard | 45-60 min |

---

## Running Solutions

```bash
cd exercises/solutions
python 01_dropout.py
```

Each solution includes test functions that verify correctness and print results.

---

## Key Differences from Skeletons

**Exercise 1 (Dropout):** The mask is scaled in forward, so backward just multiplies by mask.

**Exercise 2 (Layer Norm):** Use `keepdims=True` for proper broadcasting. The backward pass is mathematically complex.

**Exercise 3 (Weight Decay):** The 0.5 factor makes the gradient exactly `lambda * W`.

**Exercise 4 (Early Stopping):** Reset counter on ANY improvement, even tiny ones.

**Exercise 5 (Full Pipeline):** Validation must use `training=False` for dropout.

---

## Common Mistakes Checklist

- [ ] Dropout applied at test time (should be OFF)
- [ ] Missing 1/keep_prob scaling (breaks expected value)
- [ ] Wrong axis for layer norm (should be -1)
- [ ] Missing epsilon in layer norm (division by zero)
- [ ] Weight decay on biases (should be weights only)
- [ ] Not resetting counter on improvement
- [ ] Dropout during validation (should be off)

---

## Going Further

After completing the exercises:

1. Add dropout between LSTM layers (not just on hidden states)
2. Try different norm types — batch norm, group norm
3. Implement dropout variants — DropConnect, spatial dropout
4. Apply to real data — PTB dataset, text classification

---

## Additional Resources

- [../README.md](../README.md) — Implementation guide with code snippets
- [../CHEATSHEET.md](../CHEATSHEET.md) — Quick formulas and hyperparameters
- [../paper_notes.md](../paper_notes.md) — Detailed paper walkthrough
- [../implementation.py](../implementation.py) — Production-quality code

---

*The real growth happens when you're stuck and figure it out. Use these solutions as a guide, not a shortcut.*
