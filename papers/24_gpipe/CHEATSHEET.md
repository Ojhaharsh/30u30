# Day 24 Cheat Sheet: GPipe

## The Big Idea (30 seconds)

GPipe splits a single large model into **partitions** ($K$) and the input batch into **micro-batches** ($M$). This "pipelines" the execution so that multiple devices are active simultaneously. To save memory, it uses **re-materialization** (checkpointing), which throws away activations during the forward pass and recomputes them during the backward pass.

**The Pipeline Visualization:**
```text
GPU 0: [M0][M1][M2][M3]
GPU 1:     [M0][M1][M2][M3]
GPU 2:         [M0][M1][M2][M3]
GPU 3:             [M0][M1][M2][M3]
      |--- Time Steps --->
```

---

## Quick Start

```bash
cd papers/24_gpipe

# Run benchmark with 4 GPUs (simulated) and 16 micro-batches
python train_minimal.py --partitions 4 --micro-batches 16 --layers 40

# Run with custom hidden dimension and activation checkpointing enabled
python train_minimal.py --hidden-dim 1024 --layers 60
```

---

## Key Parameters for Scaling

| Parameter | Typical Range | What It Does | Tips |
|-----------|---------------|--------------|------|
| `K` (Partitions) | 2 to 16 | Number of GPUs/Devices. | Match your hardware count. |
| `M` (Micro-batches) | 4x to 8x $K$ | Slices of the mini-batch. | Increasing M reduces the bubble but adds overhead. |
| `use_checkpoint` | True / False | Activation re-computation. | Set to True for giant models; False for speed. |
| `Accumulation` | 1 | Gradient summing. | GPipe is synchronous; accumulation is automatic over M. |

---

## Common Issues & Fixes

### 1. Out of Memory (OOM) at Partition Boundaries
```python
# Problem: Even with checkpointing, partition boundary activations are stored.
# Fix: Reduce the micro-batch size (N/M).
# Example: If batch size N=128 and M=4, micro-batch=32. Try M=8 (micro-batch=16).
```

### 2. High "Bubble" Overhead (Low GPU Utilization)
```python
# Problem: GPUs are sitting idle at the start/end of the pipeline.
# Fix: Increase M relative to K.
# Bubble Rate = (K-1) / (M + K - 1). 
# If K=4, M=4 -> 3/7 (42% idle). If K=4, M=32 -> 3/34 (8% idle).
```

### 3. Gradient Inconsistency
```python
# Problem: Loss doesn't match standard SGD.
# Fix: Ensure optimizer.step() is called only AFTER all M micro-batches finish.
# GPipe should accumulate gradients across all M steps before updating weights.
```

---

## The Math (Copy-Paste Ready)

### Pipeline Efficiency
```python
def calculate_bubble_overhead(k_partitions, m_microbatches):
    """Ref: Section 3.1.2 Bubble Rate."""
    return (k_partitions - 1) / (m_microbatches + k_partitions - 1)
```

### Memory Scaling Law
```python
def estimate_memory_reduction(k_partitions, m_microbatches, layers, batch_size):
    """
    Standard: O(BatchSize * Layers)
    GPipe: O(BatchSize * Layers / K + (BatchSize / M) * Layers)
    """
    standard = batch_size * layers
    gpipe = (batch_size * layers / k_partitions) + (batch_size / m_microbatches * layers)
    return standard / gpipe
```

---

## Visualization Quick Reference

To visualize the synchronous schedule, we track the execution of each `(microbatch_idx, stage_idx)` pair.

| Step | Device 0 | Device 1 | Device 2 | Note |
|------|----------|----------|----------|------|
| 0 | MB 0 | Idle | Idle | Fill start |
| 1 | MB 1 | MB 0 | Idle | |
| 2 | MB 2 | MB 1 | MB 0 | **Perfect Steady State** |
| 3 | Idle | MB 2 | MB 1 | Drain start |

---

## Success Criteria (10/10 Verification)

- **Equivalence**: Gradients produced by GPipe must be identical to a sequential model (within float tolerance).
- **Memory**: Peak memory usage should be significantly lower when `use_checkpoint=True`.
- **Utilization**: Hardware traces should show the "Pipeline Bubble" characteristic at the start and end of batches.

---

## Experiment Ideas

| Tier | Idea | What to look for |
|------|------|------------------|
| **Easy** | Sensitivity to M | Plot speedup vs. number of micro-batches (M=1 to M=32). |
| **Medium** | Checkpointing Overhead | Measure the exact time penalty of re-computation vs. memory gap. |
| **Hard** | Non-uniform Partitioning | Try splitting layers unevenly (e.g., 80% on Device 0). Watch utilization crash. |

---

## File Reference

| File | Use It For |
|------|------------|
| `implementation.py` | Building your own GPipe wrapper or learning partition logic. |
| `visualization.py` | Generating Gantt charts for reports or deep understanding. |
| `setup.py` | Running safety checks to ensure math correctness. |
| `paper_notes.md` | Preparing for interviews/exams or internalizing the theory. |

---

**Next: Day 25 - Scaling Laws for Neural Language Models**
