# Day 6 Cheat Sheet: The First Law of Complexodynamics

## The Big Idea (30 seconds)

Entropy goes up monotonically. "Complexity" (interestingness) peaks at intermediate times, then drops back down. Aaronson proposes **complextropy** — resource-bounded sophistication — as the formal measure that captures this. The conjecture: complextropy is small at early times, large at intermediate times, small at equilibrium. Unproven as of 2024.

**The one picture to remember:**

```
complexity
    ^
    |        *  *  *
    |      *          *
    |    *              *
    |  *                  *
    | *                      *
    +--*---------------------*--> time
    t=0    tendrils      equilibrium
```

---

## Quick Start

```bash
cd papers/06_Complexodynamics

# Run coffee mixing simulation with complexity tracking
python train_minimal.py --grid-size 64 --steps 50000

# Compare different complexity measures
python train_minimal.py --grid-size 64 --steps 50000 --compare-measures

# Multi-scale analysis
python train_minimal.py --grid-size 64 --steps 50000 --multiscale

# Generate all visualizations
python visualization.py
```

---

## Complexity Measures at a Glance

| Measure | What It Is | Pros | Cons |
|---------|-----------|------|------|
| **Kolmogorov complexity (KC)** | Shortest program producing x | Theoretically clean | Uncomputable; random = max KC |
| **gzip size** | Compressed byte length | Fast, reproducible | Not optimal; misses complex patterns |
| **Sophistication** | KC of smallest "model" S containing x where x is random in S | Low for both simple AND random | Uncomputable in general; O(log t) for deterministic systems |
| **Complextropy** | Resource-bounded sophistication | Should peak at intermediate times | Definition is informal; conjecture unproven |
| **Coarse-grained KC** | gzip of blurred/downsampled state | Captures macro structure | Scale choice is ad hoc |
| **Two-part code** | Model description + residual | Directly approximates sophistication | No canonical model choice |
| **Logical depth** | Time for shortest program to run | Low for simple AND random | Hard to compute; debated whether it peaks (Bennett vs Aaronson) |

---

## Key Parameters for the Coffee Simulation

| Parameter | Typical Range | What It Does | Tips |
|-----------|--------------|--------------|------|
| `grid_size` | 32-128 | Side length of square pixel grid | 64 is good balance. 128 is slow but shows better detail. |
| `n_steps` | 10000-100000 | Number of random swaps | Need ~grid_size^3 to reach equilibrium |
| `compresslevel` | 9 | gzip compression level | Always use 9 for consistency |
| `block_size` | 2-16 | Coarse-graining block size | Try grid_size/8 as starting point |
| `measure_interval` | 100-1000 | Steps between complexity measurements | Too frequent = slow; too rare = miss the peak |

---

## Common Issues & Fixes

### gzip Gives Noisy Complexity Estimates

```python
# Problem: single gzip measurement is noisy
# Fix: average over multiple serialization orders or runs

def robust_gzip(grid, n_samples=5):
    sizes = []
    data = grid.astype(np.uint8).tobytes()
    sizes.append(len(gzip.compress(data, compresslevel=9)))
    # Also try column-major
    data_col = grid.T.astype(np.uint8).tobytes()
    sizes.append(len(gzip.compress(data_col, compresslevel=9)))
    return np.mean(sizes)
```

### Complexity Curve Is Flat (No Hump)

```python
# Problem: grid too small — structure is too coarse to measure
# Fix 1: increase grid_size to at least 64
grid_size = 64  # not 16

# Fix 2: measure more frequently in the early phase
# Most of the action happens in the first 30% of steps
measure_points = np.concatenate([
    np.linspace(0, n_steps * 0.3, 100),
    np.linspace(n_steps * 0.3, n_steps, 50)
]).astype(int)
```

### Simulation Is Too Slow

```python
# Problem: swapping one pair per step is very slow
# Fix: vectorize — swap multiple non-adjacent pairs per step
def batch_swap(grid, n_swaps=100):
    """Swap n_swaps random adjacent coffee-milk pairs per step."""
    rows, cols = grid.shape
    for _ in range(n_swaps):
        r, c = np.random.randint(0, rows), np.random.randint(0, cols-1)
        if grid[r, c] != grid[r, c+1]:
            grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c]
    return grid
```

### Coarse-Graining Produces All Zeros or All Ones

```python
# Problem: block_size too large relative to grid
# Fix: ensure block_size << grid_size
assert block_size <= grid_size // 4, (
    f"block_size {block_size} too large for grid_size {grid_size}"
)
```

---

## The Math (Copy-Paste Ready)

### Kolmogorov Complexity (KC)

```python
# KC is uncomputable. gzip gives an upper bound.
import gzip

def kc_approx(data: bytes) -> int:
    """Upper bound on KC via gzip compression."""
    return len(gzip.compress(data, compresslevel=9))
```

### Shannon Entropy of a Grid

```python
import numpy as np

def grid_entropy(grid):
    """Shannon entropy of pixel distribution."""
    p = grid.mean()  # fraction of 1s (coffee)
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
```

### Coarse-Grained KC

```python
def coarse_grained_kc(grid, block_size):
    """KC of blurred grid (Carroll's measure)."""
    h, w = grid.shape
    coarse = np.zeros((h // block_size, w // block_size))
    for i in range(coarse.shape[0]):
        for j in range(coarse.shape[1]):
            block = grid[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]
            coarse[i, j] = block.mean()
    # Quantize to uint8 for compression
    quantized = (coarse * 255).astype(np.uint8)
    return kc_approx(quantized.tobytes())
```

### Two-Part Code (Sophistication Proxy)

```python
def two_part_code(grid, block_size):
    """
    Part 1: coarse model (sophistication proxy)
    Part 2: residual (randomness)
    """
    h, w = grid.shape
    coarse = np.zeros((h // block_size, w // block_size))
    for i in range(coarse.shape[0]):
        for j in range(coarse.shape[1]):
            block = grid[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]
            coarse[i, j] = block.mean()

    # Part 1: model description
    model_bytes = (coarse * 255).astype(np.uint8).tobytes()
    part1 = kc_approx(model_bytes)

    # Part 2: residual (reconstruct - actual)
    reconstructed = np.repeat(np.repeat(coarse, block_size, axis=0),
                              block_size, axis=1)[:h, :w]
    residual = (grid - reconstructed)
    residual_bytes = ((residual + 1) * 127).astype(np.uint8).tobytes()
    part2 = kc_approx(residual_bytes)

    return part1, part2
```

---

## Visualization Quick Reference

```python
import matplotlib.pyplot as plt

# Plot complexity over time (the "hump")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: entropy (monotone increase)
axes[0].plot(times, entropies)
axes[0].set_title("Entropy (monotone)")
axes[0].set_xlabel("Time (swaps)")

# Right: complexity (peaks then drops)
axes[1].plot(times, complexities)
axes[1].set_title("Complexity (the hump)")
axes[1].set_xlabel("Time (swaps)")

plt.tight_layout()
plt.savefig("entropy_vs_complexity.png", dpi=150)
```

---

## Key Relationships

```
KC(x) = high   --> x could be random OR structured
KC(x) = low    --> x is simple (compressible)

soph(x) = high --> x is genuinely structured (not random)
soph(x) = low  --> x is simple OR random

complextropy(x, t) = high --> x is structured AND arose from a
                              time-bounded physical process
```

---

## Blog Comment Highlights (Worth Reading)

| Who | Comment # | Key Point |
|-----|----------|-----------|
| Sean Carroll | #6-7 | Coarse-graining is essential; complexity might not be smooth |
| Charles Bennett | #110 | Argues FOR logical depth; intermediate states are deep because you must simulate the mixing |
| Aaronson response | #112 | Skeptical of depth; reports Ouellette's results showing the hump with gzip |
| Luca Trevisan | #17 | Multi-scale approach: complexity lives at intermediate scales too |
| Abram Demski | #47 | Argues depth increases linearly at first if rules are "interesting" |
