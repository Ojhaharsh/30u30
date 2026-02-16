# MSI Exercises

Practice implementing and understanding universal intelligence.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Weighted Reward Summation | Easy (2/5) | 20 min |
| 2 | Designing Environments | Medium (3/5) | 30 min |
| 3 | Epsilon-Greedy Intelligence | Easy (2/5) | 15 min |
| 4 | Pattern Recognition Agent | Medium (3/5) | 40 min |
| 5 | Complexity Invariance | Hard (4/5) | 60 min |

---

## Exercise 1: Weighted Reward Summation

**Goal:** Implement the core $\Upsilon(\pi) = \sum 2^{-K(\mu)} V_\mu^\pi$ calculation.
**Tasks:**
- Open `exercise_01_formula.py`.
- Complete the `calculate_upsilon` function.
- Verify that performance on simpler tasks contributes more to the final score than performance on complex ones.

## Exercise 2: Designing Environments

**Goal:** Create a new environment with a specific Kolmogorov complexity.
**Tasks:**
- Open `exercise_02_environments.py`.
- Build a "Binary Search" environment.
- Assign it a complexity score $K$ based on the number of constraints it has.

## Exercise 3: Epsilon-Greedy Intelligence

**Goal:** Analyze how exploration ($\epsilon$) affects universal intelligence.
**Tasks:**
- Run the provided script with different $\epsilon$ values.
- Observe the trade-off between score on simple vs. complex tasks.

## Exercise 4: Pattern Recognition Agent

**Goal:** Implement an agent that uses simple history to predict future rewards.
**Tasks:**
- Open `exercise_04_agent.py`.
- Fill in the `HistoryAgent` logic.
- Compare its $\Upsilon$ score against a `RandomAgent`.

---

## How to Use

1.  Read the exercise file — each has detailed instructions in the docstrings.
2.  Find the `TODO` sections — these are what you need to implement.
3.  Run the tests — each file has a `if __name__ == "__main__":` test block.
4.  Check solutions — if you get stuck, compare your work with `solutions/solution_X.py`.

## Tips

-   Keep your environment rewards bounded in $[0, 1]$.
-   Remember that $2^{-K}$ drops off very fast; complexity $K > 10$ will have almost zero impact unless the reward is massive.
-   Universal intelligence is about **generalization**, not mastery of a single task.

---

**Next:** [Day 28 — CS231n: CNNs for Visual Recognition](../28_cs231n/)
