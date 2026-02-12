# GPipe Exercise Solutions

This directory contains the reference implementations for the Day 24 exercises. 

> [!TIP]
> Try to implement these yourself in the parent `exercises/` directory before looking at the solutions. Pipeline parallelism is nuanced, and getting "stuck" on the micro-batch logic is a key part of the learning process.

## Solution Index

### 1. [Micro-batch Splitting](solution_01_microbatch.py)
Uses `torch.chunk` to slice the batch dimension. Shows how to handle cases where the batch size isn't perfectly divisible by the number of micro-batches (though divisibility is preferred).

### 2. [Pipeline Forward Pass](solution_02_pipeline_forward.py)
Implements the synchronous clock-cycle schedule. This is the "heart" of GPipe, ensuring that multiple partitions are filled and drained efficiently.

### 3. [Gradient Accumulation](solution_03_gradients.py)
Demonstrates how to manually accumulate gradients from micro-batches. Key takeaway: GPipe is synchronous, so we sum micro-batch gradients before the optimizer step.

### 4. [Re-materialization](solution_04_checkpoint.py)
Wraps a sequence of layers in `torch.utils.checkpoint`. This is the memory-saving champion of the paper, trading a small compute overhead for a massive reduction in activation storage.

### 5. [Full GPipe Wrapper](solution_05_full_gpipe.py)
The final integration. Combines partitioning, scheduling, and checkpointing into a single `nn.Module` that behaves like a standard sequential model.

---

## Verification
You can verify these solutions by running them directly (they include minimal test inputs) or by running:
```bash
python setup.py
```
This script runs a suite of mathematical equivalence tests against the finalized implementations.
