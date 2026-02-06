# Day 7: Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton

> Scott Aaronson, Sean M. Carroll, Lauren Ouellette (2014) — [Paper](https://arxiv.org/abs/1405.6903)

**Time:** 3-4 hours
**Prerequisites:** Days 4-6 helpful (Kolmogorov complexity, MDL, complexodynamics)
**Code:** Python + NumPy + gzip + Matplotlib

---

## What This Paper Is Actually About

This paper is the empirical follow-up to Day 6. Aaronson's blog post asked: can we define a complexity measure that goes up then back down while entropy goes monotonically up? This paper *runs the experiment*.

Lauren Ouellette (then an MIT undergrad — the one Aaronson mentioned in the Day 6 blog post as "a wonderful MIT undergrad...recently started a research project with me") implemented a two-dimensional coffee-mixing cellular automaton in Python, measured its "apparent complexity" over time using gzip compression of coarse-grained bitmaps, and showed: yes, complexity rises and falls while entropy only increases.

The paper does three things:

1. **Reviews four complexity measures** — apparent complexity, sophistication, logical depth, and light-cone complexity — and explains why they chose apparent complexity for the experiment (the others are uncomputable or can't be efficiently approximated).

2. **Defines a concrete model** — the "coffee automaton," an N x N binary grid where cream pixels (1) and coffee pixels (0) mix by random adjacent swaps.

3. **Runs the experiment** — measures both entropy (gzip of fine-grained state) and apparent complexity (gzip of coarse-grained state) over time, for both an "interacting" model (particles can't overlap — the main one) and a "non-interacting" model (cream particles random-walk independently).

The key empirical result (Figures 2, 10): for the interacting model, complexity rises then falls. For the non-interacting model, complexity stays flat (once thresholding artifacts are removed). This difference matters — it says that *interactions* are what create macroscopic complexity.

The paper also proves analytically (Appendix, Section 9) that the non-interacting model's apparent complexity never becomes large, using Chernoff bounds on the expected cream distribution.

---

## The Core Idea

The central hypothesis from Day 6, now tested experimentally:

```
                    Apparent Complexity
                          ^
                         /|\
                        / | \
                       /  |  \
                      /   |   \
            ---------/    |    \---------
           /              |              \
     ─────                |                ─────
     t=0             t_peak              t→ infinity

     Entropy: ─────────────────────────────/────
                  (monotonically increasing)
```

"Apparent complexity" = the Kolmogorov complexity of a smoothed (coarse-grained) version of the state. In practice: blur the bitmap by averaging over local squares, threshold into discrete buckets, and compress with gzip. The compressed size is the complexity estimate.

**Why coarse-graining works**: A totally ordered state (all coffee on bottom, all cream on top) compresses well at any scale. A totally random state (uniform mixture) compresses well at the coarse-grained level because averaging turns local randomness into a uniform gray. But in between — when you have tendrils and boundaries — the coarse-grained representation has genuine, incompressible structure.

---

## What the Authors Actually Showed

### The Coffee Automaton Model (Section 3)

Two models on an N x N binary grid:

**Interacting model** (Section 3.1): The main model. At each time step, pick a random pair of horizontally or vertically adjacent pixels that are different colors, and swap them. Particles can't overlap — one per cell.

**Non-interacting model** (Section 3.2): Each cream particle independently random-walks to a neighbor at each step. Multiple particles can occupy the same cell. Only cream particles are tracked; coffee is just the background.

Initial state for both: top half = cream (1), bottom half = coffee (0).

```python
# The interacting model in pseudocode (from Section 3.1):
# At each time step:
#   1. Pick a random adjacent pair (horizontally or vertically)
#   2. If they differ, swap them
#   3. That's it. One swap per step.
```

The interacting model is more physically realistic — two particles can't occupy the same point. But the non-interacting model is easier to analyze theoretically because each particle is an independent random walk.

### Approximating Apparent Complexity (Section 4)

The paper tried three approaches to estimate complexity:

1. **OSCR algorithm** (Evans et al. 2003) — a two-part code that directly estimates K(S) and K(x). "We implemented a version of this algorithm, and we found that our implementation does not perform well in compressing the automaton data. The output of the algorithm is noisy." (Section 4) *Rejected.*

2. **Two-part code via coarse-graining** — coarse-grained state as Part 1, diff-to-fine-grained as Part 2. "Our estimate of K(x|S) suffered from artifacts." (Section 4) *Rejected.*

3. **Direct coarse-graining** — compress the coarse-grained state directly with gzip. This worked. Entropy is estimated by gzip of the fine-grained state; complexity by gzip of the coarse-grained state.

The coarse-graining procedure (Section 5.1):
- Divide the N x N grid into g x g squares (g = grain size)
- Average the values in each square
- Threshold the floating-point averages into discrete buckets (3 buckets in Section 5, 7 buckets in Section 6)
- Compress the thresholded array with gzip
- The compressed size = estimated apparent complexity

### Coarse-Graining Experiment (Section 5)

Method: 3-bucket thresholding (areas that are mostly coffee, mostly cream, or mixed).

Results (Figure 2):
- **Interacting model**: Complexity rises then falls. Entropy increases monotonically. The complexity peak occurs at the time when there are interesting macroscopic structures (tendrils of cream reaching into coffee and vice versa).
- **Non-interacting model**: Also showed rising-then-falling complexity, BUT this turns out to be an artifact of the thresholding (border pixels fluctuating between bucket values).

Scaling behavior (Figures 6-8):
- Max entropy scales as n^2 (quadratic — proportional to number of particles)
- Max complexity scales as n (linear — proportional to side length)
- Time to max complexity scales as n^2 (quadratic — proportional to particle count)

The linear scaling of max complexity is notable: complexity develops "along a single dimension of the automaton" (the horizontal boundary between coffee and cream).

### Adjusted Coarse-Graining (Section 6)

To address thresholding artifacts, the authors:
- Used 7 buckets instead of 3
- Applied a row-majority adjustment: if a cell is within one threshold of the majority value in its row, snap it to the majority value

Results (Figure 10):
- **Interacting model**: Still shows the rise-and-fall pattern. The complexity curve is preserved.
- **Non-interacting model**: Complexity flattened to near zero. The previous rise-and-fall was entirely a thresholding artifact.

This is the paper's most important empirical finding: **complexity requires interaction**. Non-interacting particles produce no macroscopic structure because each is an independent random walk. The expected distribution can be predicted from the initial conditions plus time, so the coarse-grained state has low KC at all times.

### Analytical Result (Section 9, Appendix)

For the non-interacting model with periodic boundary conditions, the authors prove:

The expected number of cream particles at position (x,y) satisfies:

$$E[a_{t+1}(x,y)] = \frac{E[a_t(x-1,y)] + E[a_t(x+1,y)] + E[a_t(x,y-1)] + E[a_t(x,y+1)]}{4}$$

Using Chernoff bounds, they show that in any L x L square B, the actual count stays close to the expected count with high probability, provided the grain size L >> G * sqrt(3 * ln(2n^2)). This means the coarse-grained image can be reconstructed from n and t alone, so its KC is at most O(log n + log t).

### What They Chose NOT to Measure (Section 2)

This matters pedagogy-wise:

- **Sophistication**: "sophistication as defined above seems irrelevant to the coffee cup or other physical systems: it simply never becomes large for such systems!" (Section 2.2). With overwhelming probability, the output of a short probabilistic program has low sophistication.
- **Logical depth**: "generating what many people would regard as a visually complex pattern...simply need not take a long time!" (Section 2.3). Also "even less clear how to estimate it in practice."
- **Light-cone complexity**: Interesting but requires knowing the entire causal history, not just the current state.

They chose **apparent complexity** for a simple reason: "we did not know of any efficient way to approximate sophistication or logical depth." (Section 2.5)

---

## Implementation Notes

### Key Decisions

**The swap rule matters**: One swap per step in the interacting model meant the original paper needed tens of millions of steps for a 100x100 grid. Our implementation uses batch swaps (multiple non-conflicting swaps per step) to speed this up while preserving the qualitative behavior. The paper's source code was written in Python by Lauren Ouellette.

**gzip level consistency**: Use the same compression level throughout. The paper showed (Figure 5) that different compressors (gzip, bzip2, etc.) produce qualitatively similar curves. We use `gzip.compress(data, compresslevel=9)`.

**Grain size selection**: The paper doesn't specify exactly how they chose g. Too small and the coarse-graining doesn't smooth out noise. Too large and it erases all structure. A reasonable default is g = n/10 (for a 100x100 grid, g = 10).

**Border pixel artifacts**: The whole point of Section 6 was to fix these. The 3-bucket version showed fake complexity in the non-interacting model. The 7-bucket + majority adjustment fixed it. Our implementation includes both versions.

### Things That Will Bite You

1. **gzip of binary data, not strings**: Convert the numpy array to bytes before compressing. `grid.tobytes()` or saving as a bitmap format.

2. **Thresholding**: After averaging a g x g block, you get floats. Threshold into discrete buckets BEFORE compressing. The paper uses 3 buckets (Section 5) and 7 buckets (Section 6).

3. **Entropy is NOT Shannon entropy here**: The paper uses "entropy" to mean gzip-compressed size of the fine-grained state, not the standard information-theoretic formula. This is a KC proxy.

4. **Non-interacting ≠ non-existent particles**: In the non-interacting model, cream particles overlap. This means you need a count grid (uint16 or higher), not a binary grid.

5. **Scale of time steps**: For an N x N interacting grid, expect O(N^4) single-swap steps to reach equilibrium. With batch swaps, this comes down substantially, but it's still slow for large grids.

---

## What to Build

### Quick Start

```bash
python train_minimal.py --grid-size 50 --steps 100000 --model interacting
```

This runs the coffee automaton, measures entropy and apparent complexity at each snapshot, and plots the rise-and-fall curve.

More options:
```bash
# Compare interacting vs non-interacting
python train_minimal.py --grid-size 50 --steps 100000 --compare-models

# Run the adjusted coarse-graining (Section 6)
python train_minimal.py --grid-size 50 --steps 100000 --adjusted

# Scaling experiment: how does max complexity scale with grid size?
python train_minimal.py --scaling --sizes 20 30 50 70 100
```

### Exercises

Exercises are in `exercises/`. Each one is self-contained.

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Coarse-graining implementation (`exercise_01_coarse_graining.py`) | Build the paper's core measurement tool from scratch |
| 2 | Interacting vs non-interacting (`exercise_02_model_comparison.py`) | See why interaction matters for complexity |
| 3 | Scaling analysis (`exercise_03_scaling.py`) | Replicate Figures 6-8: max complexity ~ n, max entropy ~ n^2 |
| 4 | Adjusted coarse-graining (`exercise_04_adjusted_graining.py`) | Implement Section 6's artifact-removal technique |
| 5 | Compressor comparison (`exercise_05_compressor_comparison.py`) | Replicate Figure 5: different compressors, same qualitative curve |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **Apparent complexity = KC of the coarse-grained state.** Smooth out noise by averaging local regions, threshold, compress with gzip. This is computable, reproducible, and captures the intuitive notion of "interestingness." (Sections 2.1, 4)

2. **Complexity rises then falls; entropy only rises.** For the interacting coffee automaton, gzip(coarse-grained) peaks at intermediate times while gzip(fine-grained) increases monotonically. This is the central empirical result. (Figures 2, 10)

3. **Interactions create complexity; independence does not.** The non-interacting model never develops genuine macroscopic complexity. Each particle's position is an independent random walk, so the expected coarse-grained state can be predicted from initial conditions alone, giving O(log n) KC. (Section 6, Appendix)

4. **Max complexity scales linearly with grid size; max entropy scales quadratically.** Complexity develops along the one-dimensional coffee-cream boundary (so ~ n). Entropy counts all particles (so ~ n^2). Time to peak complexity ~ n^2. (Figures 6-8)

5. **Thresholding artifacts are a real problem.** The naive 3-bucket coarse-graining showed fake complexity in the non-interacting model. The 7-bucket + row-majority adjustment removed it. This is why Section 6 exists and why the adjusted method is more trustworthy. (Section 6)

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Coffee automaton (interacting + non-interacting), coarse-graining, gzip KC estimation |
| `train_minimal.py` | CLI script — run simulations, measure complexity/entropy over time, plot results |
| `visualization.py` | Entropy vs complexity curves, grid state visualization, scaling plots |
| `notebook.ipynb` | Interactive walkthrough — build the automaton and measurements step by step |
| `exercises/` | 5 exercises: coarse-graining, model comparison, scaling, adjusted graining, compressors |
| `paper_notes.md` | Detailed notes on the paper's arguments and proofs |
| `CHEATSHEET.md` | Quick reference for the automaton model, measures, and key results |

---

## Further Reading

- [The Paper (arXiv:1405.6903)](https://arxiv.org/abs/1405.6903) — 22 pages, many figures, very readable
- [Day 6 Blog Post](https://scottaaronson.blog/?p=762) — Aaronson's original blog post proposing complextropy (the theoretical foundation for this paper)
- [Sophistication (Koppel 1987)](https://doi.org/10.1007/BF01200260) — the complexity-of-the-model concept that motivates apparent complexity
- [Gacs, Tromp, Vitanyi (2001)](https://doi.org/10.1109/TIT.2001.936000) — algorithmic statistics and structure functions
- [Logical Depth (Bennett 1995)](https://doi.org/10.1007/978-3-7091-6597-3_8) — the time-based alternative (discussed in Section 2.3 but not used)
- [Light-Cone Complexity (Shalizi et al. 2004)](https://doi.org/10.1103/PhysRevLett.93.118701) — the mutual information approach (discussed in Section 2.4)
- [Antunes and Fortnow (2009)](https://doi.org/10.1007/s00224-008-9106-3) — proved equivalence of coarse sophistication and Busy Beaver depth

---

**Previous:** [Day 6 — The First Law of Complexodynamics](../06_Complexodynamics/)
**Next:** [Day 8 — AlexNet](../08_alexnet/)
