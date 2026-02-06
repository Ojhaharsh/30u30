# CHEATSHEET: The Coffee Automaton

> Quick reference for Aaronson, Carroll, Ouellette (2014), arXiv:1405.6903

---

## The Model

### Setup
- N x N grid of binary values: 1 = cream, 0 = coffee
- Initial state: top half = 1 (cream), bottom half = 0 (coffee)

### Interacting Model (Section 3.1) — the main one
```
At each time step:
  1. Pick a random pair of adjacent cells (horizontally or vertically)
  2. If they are different, swap them
  3. One swap per step
```

### Non-Interacting Model (Section 3.2) — the control
```
At each time step:
  For each cream particle:
    Move to a random neighbor (up/down/left/right)
  Multiple particles can occupy one cell
  Coffee is just background
```

---

## The Measurements

### Entropy Estimate (proxy for K(x))
```python
entropy = len(gzip.compress(grid.tobytes(), compresslevel=9))
```
Compress the fine-grained (original) binary grid. Monotonically increases.

### Apparent Complexity Estimate (proxy for K(f(x)))
```python
# Step 1: Coarse-grain — average over g x g blocks
coarse = average_blocks(grid, grain_size=g)

# Step 2: Threshold into discrete buckets
# Section 5: 3 buckets (0, 0.5, 1)
# Section 6: 7 buckets (0, 1/6, 2/6, ..., 1)
thresholded = threshold(coarse, num_buckets)

# Step 3: Compress
complexity = len(gzip.compress(thresholded.tobytes(), compresslevel=9))
```
Rises then falls for interacting model. Stays flat for non-interacting (after adjustment).

### Adjusted Coarse-Graining (Section 6)
```python
# After thresholding, apply row-majority adjustment:
for each row in thresholded:
    majority_value = mode(row)
    for each cell in row:
        if abs(cell - majority_value) <= 1 bucket:
            cell = majority_value
```
Removes border pixel artifacts. Non-interacting complexity drops to near zero.

---

## Key Equations

### Apparent Complexity (Definition)
$$C_{\text{apparent}}(x) = K(f(x))$$
where $f$ = smoothing function, $K$ = Kolmogorov complexity.

### Sophistication (for theory reference — NOT computed in experiments)
$$\text{soph}_c(x) = \min\{K(S) : x \in S, \; K(S) + \log_2|S| \leq K(x) + c\}$$

### Apparent Complexity as Resource-Bounded Sophistication (Section 2.5)
$$K(f(x)) \approx K(S_{f,x}) \quad \text{where } S_{f,x} = \{y : f(y) = f(x)\}$$

### Non-Interacting Random Walk (Appendix)
$$E[a_{t+1}(x,y)] = \frac{E[a_t(x-1,y)] + E[a_t(x+1,y)] + E[a_t(x,y-1)] + E[a_t(x,y+1)]}{4}$$

### Chernoff Concentration (Appendix)
$$\Pr[|a_t(B) - E[a_t(B)]| > L^2/G] < 2\exp\left(-\frac{L^2}{3G^2}\right)$$

Grain size needed: $L \gg G\sqrt{3 \ln(2n^2)}$

---

## Scaling Results (Figures 6-8)

| Quantity | Scaling | Fit $r^2$ |
|----------|---------|-----------|
| Max entropy | ~ $n^2$ (quadratic) | 0.9999 |
| Max complexity (interacting) | ~ $n$ (linear) | 0.9798 |
| Max complexity (non-interacting) | ~ $n$ (linear) | 0.9729 |
| Time to max complexity (interacting) | ~ $n^2$ (quadratic) | 0.9878 |
| Time to max complexity (non-interacting) | ~ $n^2$ (quadratic) | 0.9927 |

Why max complexity ~ n: complexity develops along the 1D boundary between coffee and cream.
Why max entropy ~ n^2: entropy counts all n^2 particles.

---

## Quick API Reference (implementation.py)

### CoffeeAutomaton
```python
from implementation import CoffeeAutomaton

# Create automaton
ca = CoffeeAutomaton(grid_size=100, model='interacting')

# Run steps
ca.step(num_steps=1000)  # batch swap for speed

# Get current state
grid = ca.grid  # numpy uint8 array, 0 or 1

# Measure
entropy = ca.entropy()        # gzip(fine-grained)
complexity = ca.complexity(    # gzip(coarse-grained)
    grain_size=10,
    num_buckets=7,
    adjust=True
)
```

### Measurements
```python
from implementation import gzip_size, coarse_grain, threshold_array

# Raw gzip size of any data
size = gzip_size(data_bytes)

# Coarse-grain a grid
coarse = coarse_grain(grid, grain_size=10)

# Threshold into buckets
discrete = threshold_array(coarse, num_buckets=7)
```

---

## Four Complexity Measures (Section 2) — Why Only One Was Used

| Measure | What It Captures | Why Not Used |
|---------|-----------------|--------------|
| Apparent complexity | KC of smoothed state | **USED** — computable, matches intuition |
| Sophistication | Size of minimal model | Never large for short-program outputs |
| Logical depth | Time of shortest program | Complex patterns don't need long computation |
| Light-cone complexity | Past-future mutual info | Requires causal history, not just current state |

---

## Common Gotchas

1. **gzip operates on bytes, not floats**: Convert grid to bytes before compressing
2. **Threshold BEFORE compressing**: Floating-point averages won't compress well
3. **One swap per step is slow**: For N=100, need ~10^7 steps. Use batch swaps.
4. **Non-interacting needs count grid**: Particles overlap, so use uint16, not uint8
5. **"Entropy" in this paper = gzip(fine-grained)**: Not Shannon entropy formula
6. **3-bucket thresholding has artifacts**: Use 7-bucket + adjustment for reliable results
7. **Grain size matters**: Too small = noise not smoothed. Too large = structure erased. Default: g = n/10.

---

## Paper References

- Koppel (1987) — sophistication definition
- Bennett (1995) — logical depth definition
- Evans et al. (2003) — OSCR two-part code (tried but didn't work)
- Shalizi et al. (2004) — light-cone complexity
- Gacs, Tromp, Vitanyi (2001) — algorithmic statistics
- Antunes & Fortnow (2009) — coarse sophistication = Busy Beaver depth
- Gell-Mann (1994) — "The Quark and the Jaguar," informal discussion of complexity rise-and-fall
- Carroll (2010) — "From Eternity to Here," arrow of time and complexity
