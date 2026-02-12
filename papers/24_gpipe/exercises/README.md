# Exercises: GPipe Pipeline Parallelism

These exercises will guide you through building a scale-up infrastructure from scratch. You will implement micro-batch splitting, synchronous pipeline scheduling, and memory-saving checkpointing.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Micro-batch Splitting | Easy (2/5) | 15 min |
| 2 | Simple Pipelining | Medium (3/5) | 30 min |
| 3 | Gradient accumulation | Medium (3/5) | 30 min |
| 4 | Re-materialization | Hard (4/5) | 45 min |
| 5 | Full GPipe Wrapper | Expert (5/5) | 60 min |

## Exercise 1: Micro-batch Splitting
Implementation of the logic to divide a mini-batch into $M$ chunks. 
- **File**: `exercise_01_microbatch.py`
- **Goal**: Correctly handle tensor splitting while ensuring reproducibility.

## Exercise 2: Simple Pipelining
Logic for the forward pass where stages are processed in sequence.
- **File**: `exercise_02_pipeline_forward.py`
- **Goal**: Coordinate the transfer of outputs from stage $k$ to stage $k+1$.

## Exercise 3: Gradient Accumulation
Ensure that the backward pass correctly accumulates gradients across all micro-batches before the optimizer step.
- **File**: `exercise_03_gradients.py`
- **Goal**: Verify that pipeline gradients match non-pipeline gradients.

## Exercise 4: Re-materialization (Section 3.2)
Implement a simple wrapper that throws away internal activations and recomputes them during the backward pass.
- **File**: `exercise_04_checkpoint.py`
- **Goal**: Trade compute for memory using `torch.utils.checkpoint`.

## Exercise 5: Full GPipe integration
Combine all previous components into a production-ready class.
- **File**: `exercise_05_full_gpipe.py`
- **Goal**: A functional model wrapper that supports any `nn.Sequential` input.

## How to Use
1. Read the instructions in each file.
2. Implement the `TODO` sections.
3. Run the file to trigger the internal test functions.
4. Check solutions in `solutions/` if you get stuck.

---
**Next: Day 25 - Scaling Laws for Neural Language Models**
