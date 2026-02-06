# Exercises: Neural Turing Machines

These exercises focus on the mechanics of differentiable addressing and memory updates.

## Exercise 1: Content-based Addressing
Implement the content-based weighting formula. Given a key $k$ and a memory matrix $M$, calculate the cosine similarity and applying the softmax with strength $\beta$.

## Exercise 2: Circular Convolution (The Shift)
NTMs move their focus by shifting. Implement a circular convolution that takes a weighting $w$ and a shift vector $s$ (size 3) and returns the shifted weighting.

## Exercise 3: Memory Erase and Add
In implementation.py, the write operation is split into Erase and Add. Implement a function that takes the current memory, a weighting, an erase vector, and an add vector, and returns the updated memory.

## Exercise 4: Controller Input Concatenation
The NTM controller needs to see both the external input AND the previous read vectors. Implement the logic to concatenate the input $x$ with $n$ read vectors from the previous time step.

## Exercise 5: Sharpening Mechanism
Why do we need sharpening? Implement the sharpening formula $w^\gamma / \sum w^\gamma$ and observe what happens to a blurry weighting as $\gamma$ increases from 1 to 5.
