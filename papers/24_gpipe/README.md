# Day 24: GPipe: Efficient Training of Giant Neural Networks

> Huang et al. (2018/2019) - [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

**Time:** 6-8 hours
**Prerequisites:** Model Parallelism basics, PyTorch `nn.Sequential`, Gradient accumulation intuition
**Code:** PyTorch + Simulated Multi-Device Pipeline

---

## What This Paper Is Actually About

GPipe addresses the "Memory Wall" in Deep Learning. By 2018, the industry had hit a ceiling: models were getting larger than the memory of a single accelerator (GPU/TPU), but traditional scaling methods like **Data Parallelism** didn't help with model size—they only helped with training speed by replicating the model.

If a model has 6 billion parameters, it simply won't fit on a 16GB V100. The naive solution is **Model Parallelism**: put the first half of the layers on GPU 0 and the second half on GPU 1. However, this creates a massive efficiency problem: while GPU 1 is processing, GPU 0 is idle, and vice-versa. This is the **Pipeline Bubble** problem.

GPipe introduces a library-level solution that combines **Pipeline Parallelism** with **Micro-batching**. By splitting a mini-batch into smaller micro-batches ($M$), GPU 0 can start working on the second micro-batch while GPU 1 is still working on the first. This "fills the pipeline" and keeps hardware utilization high.

Furthermore, GPipe pioneered the use of **Re-materialization** (activation checkpointing) at the pipeline level. Instead of storing all intermediate activations for the backward pass, it only stores them at partition boundaries, re-computing the rest on the fly. This allows researchers to train models that are 25x larger than what would otherwise fit on the same hardware.

---

## The Core Idea

The central innovation is the **Synchronous Pipeline Schedule**. In a standard model parallel setup, only one device is active at a time. GPipe turns this sequential process into a parallel pipeline by slicing the input data.

```text
Simple Model Parallelism (Inefficient):
T0: GPU 0 [Batch] -> GPU 1 [Idle]
T1: GPU 0 [Idle]  -> GPU 1 [Batch]
Utilization: 50%

GPipe Pipeline Parallelism (Efficient):
T0: GPU 0 [MB 0] -> GPU 1 [Idle]
T1: GPU 0 [MB 1] -> GPU 1 [MB 0]
T2: GPU 0 [MB 2] -> GPU 1 [MB 1]
Utilization: 90%+ (depending on M)
```

By choosing a large number of micro-batches ($M$) relative to the number of partitions ($K$), the time spent waiting for the "bubble" to fill/empty becomes negligible.

---

## What the Authors Actually Showed

### 1. Scaling to Giant Models
The authors demonstrated the training of a **6-billion parameter Transformer** model. To put this in perspective for 2019, this was a massive leap. They scaled this across 8 accelerator partitions, showing that GPipe could handle models that were literally impossible to train using standard methods.

### 2. Efficiency Claims
The research showed that the bubble size $(K-1)/(M+K-1)$ effectively vanishes as $M$ increases. For a 4-partition system with 32 micro-batches, the idle time is only ~8%.

### 3. Record-Breaking Performance on ImageNet
Using GPipe, they trained **AmoebaNet-B** with 557 million parameters. This led to a then-state-of-the-art **84.4% Top-1 accuracy** on ImageNet. The paper argues that larger models aren't just a technical flex—they lead to better generalization and accuracy.

---

## The Architecture

GPipe partitions a sequential model into $K$ stages. Each stage is assigned to a physical device.

1. **Partitioning Strategy**: The model is split into $K$ cells. To maintain balance, weights are divided such that each stage has roughly equal computation time.
2. **Micro-batching Logic**: The input mini-batch $N$ is divided into $M$ equal micro-batches.
3. **The Forward Pass**:
   - Micro-batch $i$ is processed by Stage $j$.
   - Output is passed to Stage $j+1$.
   - Stage $j$ immediately picks up Micro-batch $i+1$.
4. **The Backward Pass**:
   - Updates are **Synchronous**. No parameters are updated until ALL micro-batches have completed their backward pass.
   - This ensures that GPipe is mathematically identical to standard training (no "stale gradients").

---

## Implementation Notes

The implementation in `implementation.py` simulates this behavior using PyTorch.

### Re-materialization (Section 3.2)
This is the most critical implementation detail for memory savings.
- Normally, during the forward pass, you store every single activation for every layer to use in the backward pass.
- In GPipe, we only store the activations at the **Stage Boundaries**.
- When the backward pass reaches Stage $K$, it re-runs the forward pass for that stage's internal layers to "re-materialize" the missing activations.
- **Cost**: ~33% more computation.
- **Gain**: Memory usage drops from $O(N \cdot L)$ to $O(N \cdot L/K)$.

### The Batch Norm Gotcha
Standard Batch Normalization calculates statistics across the whole batch. In a pipeline, you only see micro-batches. 
- **Solution**: GPipe typically requires "Ghost Batch Norm" (using stats from the micro-batch) or a specialized sync-batch-norm to avoid training instability.

---

## What to Build

### Quick Start
To run the benchmarking script and see the pipeline in action:
```bash
python train_minimal.py --partitions 4 --micro-batches 8 --layers 40
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Micro-batch Splitting (`exercise_01.py`) | Master `torch.chunk` and batch dimensionality logic. |
| 2 | Pipeline Forward Pass (`exercise_02.py`) | Implement the synchronous schedule loop. |
| 3 | Gradient Accumulation (`exercise_03.py`) | Ensure gradients from micro-batches sum correctly. |
| 4 | Re-materialization (`exercise_04.py`) | Use `torch.utils.checkpoint` to save memory. |
| 5 | Full GPipe Wrapper (`exercise_05.py`) | Build the complete system for training. |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **Model Parallelism isn't enough.** Naive splitting leaves GPUs idle. Pipeline parallelism with micro-batching is required for efficiency. (Section 3.1)
2. **Synchronicity matters.** Simple synchronous updates ensure that researchers don't have to debug the stability issues inherent in asynchronous systems like PipeDream.
3. **Re-materialization is the memory champion.** By trading a bit of compute time for memory, we can train models that are significantly wider and deeper. (Section 3.2)
4. **The "Bubble" is manageable.** As long as $M \gg K$, the overhead of pipelining is small enough to be ignored.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Core `GPipe` and `PipelineStage` implementation. |
| `train_minimal.py` | CLI tool for benchmarking speed vs. memory. |
| `visualization.py` | Gantt chart generator and memory scaling plots. |
| `setup.py` | Mathematical verification via unit tests. |
| `paper_notes.md` | Deep dive into the GPipe theory and experimental results. |
| `CHEATSHEET.md` | Quick reference for pipeline sizing and common issues. |
| `notebook.ipynb` | Interactive walkthrough of the pipeline schedule. |

---

## Further Reading

- [Huang et al. (2018)](https://arxiv.org/abs/1811.06965) - The original GPipe paper.
- [Google AI Blog: GPipe](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html) - High-level overview.
- [PipeDream (SOSP 2019)](https://arxiv.org/abs/1806.03377) - The asynchronous alternative for comparison.
- [Torchgpipe Design](https://arxiv.org/abs/2004.09910) - Deep dive into implementing this in PyTorch.

---

**Previous:** [Day 23 - Variational Lossy Autoencoder](../23_variational_lossy_autoencoder/)
**Next:** [Day 25 - Scaling Laws for Neural Language Models](../25_scaling_laws/)
