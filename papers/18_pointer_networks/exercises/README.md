# Exercises: Pointer Networks

These exercises are designed to help you implement the core mechanics of Pointer Networks (Vinyals et al., 2015). Focus on the relationship between the decoder hidden state (query) and the encoder hidden states (keys) to produce valid indices.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|:---|:---|:---|:---|
| 1 | Pointer Attention | Easy (2/5) | 30 min |
| 2 | Masking | Medium (3/5) | 30 min |
| 3 | Convex Hull | Medium (3/5) | 45 min |
| 4 | TSP Cost | Easy (2/5) | 20 min |
| 5 | Greedy Decoding | Hard (4/5) | 60 min |

## Exercise 1: Pointer Attention Head
Implement the additive scoring mechanism (Equation 3 from the paper). You will need to project both the encoder and decoder states, combine them, and apply the final projection to get a single score for each input position.

## Exercise 2: Masking & Constraints
Implement the masking logic required for tasks like the Traveling Salesman Problem (TSP). You must ensure that the model doesn't select the same item twice and properly ignores any padding tokens in the input sequence.

## Exercise 3: Convex Hull Formatting
Geometric problems require specific input/output formatting. Implement a function to calculate the "pointer" targets for the Convex Hull task, outputting the indices of the hull vertices in order.

## Exercise 4: TSP Cost Analysis
Write a utility function to calculate the total Euclidean distance of a TSP tour given a sequence of pointers. This reinforces the relationship between discrete pointers and the continuous geometric space they operate in.

## Exercise 5: Greedy Decoding Loop
Implement the full autoregressive decoding loop. This includes passing inputs through the decoder, selecting the best pointer, and using that selection's representation as the input for the next time step.

## How to Use

1. Read the exercise file -- each has detailed instructions.
2. Find the TODO sections -- these are what you implement.
3. Run the tests -- each file has a test function.
4. Check solutions -- compare with `solutions/solution_X.py`.

## Tips

- Use broadcasting to project the encoder states once before the decoding loop starts.
- Remember to apply the mask *before* the Softmax operation.
- In Pointer Networks, the attention distribution *is* the output layer.
