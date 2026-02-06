# Day 16 Exercises: Order Matters (Pointer Networks)

Exercises for Day 16.

These exercises will take you from basic pointer attention to a full working system for set-to-sequence problems.

---

## Exercise Structure

Each exercise builds on the previous one:

1. **Exercise 1:** Implement basic pointer attention mechanism
2. **Exercise 2:** Build order-invariant set encoder
3. **Exercise 3:** Train model to sort numbers
4. **Exercise 4:** Solve convex hull problem
5. **Exercise 5:** Tackle Traveling Salesman Problem (TSP)

---

## Learning Objectives

By completing these exercises, you will:

- Understand how pointer attention works
- Implement order-invariant encoders (sets vs sequences)
- Train models for combinatorial problems
- Visualize attention patterns
- Compare set-aware vs order-aware models

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib scipy
```

### How to Use

1. **Start with Exercise 1** - Each file has `TODO` markers
2. **Fill in the missing code** - Hints provided in comments
3. **Run your implementation** - Each file is executable
4. **Check solutions/** - When you're stuck or want to verify

### Running an Exercise

```bash
# Run exercise 1
python exercise_1.py

# With verbose output
python exercise_1.py --verbose

# Test your implementation
python exercise_1.py --test
```

---

## Exercise Difficulty

| Exercise | Difficulty | Time | Key Concepts |
|----------|------------|------|--------------|
| 1 | Easy (2/5) | 30 min | Pointer attention, argmax |
| 2 | Medium (3/5) | 45 min | Self-attention, order invariance |
| 3 | Medium (3/5) | 60 min | Training loop, loss functions |
| 4 | Hard (4/5) | 75 min | Geometric reasoning, convex hull |
| 5 | Expert (5/5) | 90 min | Combinatorial optimization, TSP |

---

## Tips

**Stuck?**
1. Check the hints in the code comments
2. Review the main README.md for concepts
3. Look at the analogies in PAPER_NOTES.md
4. Run the solution to see expected output
5. Compare your code with solutions/

**Common Issues:**
- **Attention produces NaN:** Check your masking. Inf values before softmax are fine, but check for divide-by-zero
- **Model doesn't learn:** Try smaller set sizes first (5-10 elements)
- **Memory issues:** Reduce batch size or use gradient accumulation
- **Slow training:** Use GPU if available, or reduce hidden_dim

---

## What You'll Build

By the end, you'll have:
- A working pointer network implementation
- An order-invariant set encoder
- Models trained on 3 different tasks
- Visualizations of attention patterns
- Understanding of when order matters and when it doesn't

---

## Resources

- **Paper:** https://arxiv.org/abs/1511.06391
- **Main README:** ../README.md
- **Implementation:** ../implementation.py
- **Training Script:** ../train.py

---

## Progress Tracker

Track your progress:

- [ ] Exercise 1: Basic Pointer Attention
- [ ] Exercise 2: Set Encoder
- [ ] Exercise 3: Sorting Task
- [ ] Exercise 4: Convex Hull
- [ ] Exercise 5: TSP Solver

---

Open **exercise_1.py** to begin.
