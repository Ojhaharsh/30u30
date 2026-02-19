# Exercises: RLHF

These exercises build the core components of the RLHF pipeline (Christiano et al., 2017).

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Bradley-Terry Model | Easy (2/5) | 15 min |
| 2 | Preference Loss | Medium (3/5) | 20 min |
| 3 | Reward Normalization | Easy (2/5) | 15 min |
| 4 | Synthetic Oracle | Medium (3/5) | 20 min |
| 5 | Full RLHF Loop | Hard (4/5) | 45 min |

## Exercise 1: Bradley-Terry Model
Implement the probability function that predicts if a human prefers segment A over segment B.

## Exercise 2: Preference Loss
Implement the cross-entropy loss function used to train the Reward Model.

## Exercise 3: Reward Normalization
Implement a running mean and standard deviation tracker to keep rewards stable for PPO.

## Exercise 4: Synthetic Oracle
Create a simulated "human" that labels pairs of trajectories based on the environment's ground-truth reward.

## Exercise 5: Full RLHF Loop
Stitch everything together: collect data, get labels, train the reward model, and update the policy.

## How to Use

1.  Read the exercise file — each has detailed instructions.
2.  Find the `TODO` sections — these are what you implement.
3.  Run the file — each has a test function `if __name__ == "__main__":`.
4.  Check solutions — compare with `solutions/solution_X.py`.
