# Paper Notes: Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton

> Aaronson, Carroll, Ouellette (2014) — arXiv:1405.6903

---

## ELI5

Imagine you have a glass where the top half is chocolate milk and the bottom half is regular milk. At the start, it's easy to describe: "chocolate on top, white on bottom." At the end, once they've fully mixed, it's also easy: "uniform light brown everywhere." But in the middle, when there are swirls and tendrils — that's hard to describe. You'd have to say *where each swirl is*.

This paper measures exactly how hard that middle description is, and shows it really does peak then drop back down.

> Note: This analogy is ours, not the authors'. The paper uses "coffee and cream" rather than chocolate milk, and the measurement is done computationally via compression, not verbal description.

---

## What the Paper Actually Covers

### Structure

1. **Introduction** — sets up the problem (complexity rises then falls in closed systems)
2. **Background** — reviews four complexity measures: apparent complexity, sophistication, logical depth, light-cone complexity
3. **The Coffee Automaton** — defines the two models (interacting and non-interacting)
4. **Approximating Apparent Complexity** — describes what they tried and what worked
5. **Coarse-Graining Experiment** — first experiment with 3-bucket thresholding
6. **Adjusted Coarse-Graining Experiment** — improved experiment with 7 buckets + majority adjustment
7. **Conclusions and Further Work**
8. **Appendix** — proof that non-interacting model never becomes complex

### The authors

- **Scott Aaronson** (MIT) — complexity theorist, proposed complextropy in Day 6 blog post
- **Sean M. Carroll** (Caltech) — physicist, posed the original "coffee cup" question at the FQXi conference
- **Lauren Ouellette** (MIT undergrad) — implemented the experiments. Aaronson mentioned her in the Day 6 blog post: "a wonderful MIT undergrad...recently started a research project with me"

---

## The Core Idea (From the Paper)

The paper's question (Section 1): "There is no general principle that quantifies and explains the existence of high-complexity states at intermediate times in closed systems. It is the aim of this work to explore such a principle."

Their approach:
1. Define "apparent complexity" as the KC of a coarse-grained (smoothed + thresholded) bitmap
2. Approximate KC using gzip compressed file size
3. Run the coffee automaton
4. Measure this quantity over time
5. Show it peaks at intermediate times

Why apparent complexity and not the alternatives? Direct quote (Section 2.5): "we did not know of any efficient way to approximate sophistication or logical depth."

---

## The Four Complexity Measures (Section 2)

### 2.1 Apparent Complexity

Definition: $H(f(x))$ where $H$ is an entropy measure and $f$ is a denoising/smoothing function.

In practice: $f$ = coarse-graining (average over local square, then threshold). $H$ = gzip compressed size (proxy for KC).

Advantages:
- Simple
- Computable
- Directly captures the "interestingness" intuition

Disadvantage: The smoothing function $f$ seems arbitrary. But the paper argues this is analogous to the choice of coarse-graining in Boltzmann entropy — physically motivated by the locality of interactions.

### 2.2 Sophistication (Koppel 1987)

The $c$-sophistication of string $x$:

$$\text{soph}_c(x) = \min_{S} K(S) \quad \text{subject to } x \in S, \; K(S) + \log_2|S| \leq K(x) + c$$

In words: find the smallest model $S$ that contains $x$, where $S$'s description plus the randomness needed to specify $x$ within $S$ is near-optimal.

**Why they rejected it**: For any short probabilistic program $P$ (like the coffee automaton), the output $x$ has low sophistication with overwhelming probability. Take $S$ = {all outputs $y$ of $P$ with $\Pr[y] \approx \Pr[x]$}. This $S$ takes only $O(\log n)$ bits to describe. So "sophistication as defined above seems irrelevant to the coffee cup or other physical systems: it simply never becomes large for such systems!" (Section 2.2)

However, **resource-bounded** sophistication might work — but then you need to approximate it, and they didn't know how to do that efficiently.

### 2.3 Logical Depth (Bennett)

The time taken by the shortest program that outputs $x$ (within $c$ bits of optimal).

**Why they rejected it**: "generating what many people would regard as a visually complex pattern...simply need not take a long time!" (Section 2.3). Also, "even less clear how to estimate it in practice."

### 2.4 Light-Cone Complexity (Shalizi et al.)

$$\text{LCC}(a) = I(V_{P(a)} : V_{F(a)}) = H(V_{P(a)}) + H(V_{F(a)}) - H(V_{P(a)}, V_{F(a)})$$

The mutual information between a point's past and future light-cones.

**Why they didn't use it**: It requires the causal history, not just the current state. Also, it stays large even after the system mixes (because the past light-cone contains almost the same random information as the future light-cone for slowly-changing systems).

### 2.5 Synthesis

The key insight (Section 2.5): apparent complexity is a "resource-bounded" sophistication. The coarse-graining function $f$ defines the model $S_{f,x} = \{y : f(y) = f(x)\}$, and $K(S_{f,x}) \approx K(f(x))$. So instead of minimizing over ALL possible models (which is uncomputable), you pick one specific model family (coarse-grained images) and measure its description length.

Antunes and Fortnow (2009) proved that coarse sophistication and "Busy Beaver depth" are equivalent up to $O(\log n)$.

---

## The Experiments

### What Worked and What Didn't (Section 4)

What they tried:
1. **OSCR algorithm** (Evans et al. 2003): A two-part code approach. Result: "noisy...no obvious trend." Because it "does not take into account the two-dimensionality of the automaton state."
2. **Two-part code with diff**: Coarse-grained state as Part 1, fine-to-coarse diff as Part 2. Result: "suffered from artifacts due to the way the diff was represented."
3. **Direct coarse-graining + gzip**: Compress the coarse-grained state. Entropy = gzip(fine-grained), Complexity = gzip(coarse-grained). **This worked.**

### Coarse-Graining Experiment — 3 Buckets (Section 5)

Method:
- Average values in g x g squares
- Threshold into 3 buckets: mostly coffee (near 0), mixed (near 0.5), mostly cream (near 1)
- Compress with gzip
- Complexity = compressed size of thresholded coarse-grained array
- Entropy = compressed size of fine-grained array

Results:
- Both interacting and non-interacting models show complexity rising then falling
- Entropy increases monotonically in both
- BUT the non-interacting result turns out to be an artifact

Scaling (Figures 6-8):
- Max entropy ~ $n^2$ (quadratic, proportional to particle count)
- Max complexity ~ $n$ (linear, proportional to grid side length)
- Time to max complexity ~ $n^2$

### Adjusted Coarse-Graining — 7 Buckets + Majority (Section 6)

Problem with 3 buckets: border pixel artifacts. Cells near a threshold boundary fluctuate between buckets due to small noise, creating fake complexity.

Fix:
- 7 buckets instead of 3 (finer resolution)
- Row-majority adjustment: if a cell is within 1 threshold of the majority value in its row, snap it to the majority value

Results:
- **Interacting model**: Complexity curve preserved. Rise-and-fall still appears.
- **Non-interacting model**: Complexity flattened to nearly zero. The previous rise was entirely a thresholding artifact.

This is the paper's most significant empirical finding.

### The Proof for Non-Interacting (Appendix, Section 9)

Claim: For the non-interacting model, the apparent complexity is at most $O(\log n + \log t)$ at all times.

Key steps:
1. Each cream particle does an independent random walk
2. Expected count $E[a_t(x,y)]$ depends only on vertical position (by symmetry with periodic boundary)
3. Within any $L \times L$ square $B$, actual count concentrates around expectation by Chernoff bounds
4. Provided grain size $L \gg G\sqrt{3\ln(2n^2)}$, all grains match their expected colors
5. The coarse-grained image can be reconstructed from $n$ and $t$ alone
6. Therefore KC is $O(\log n + \log t)$

---

## The Math

### Entropy Estimates

**Kolmogorov complexity** (used as the theoretical foundation):
$$K(x) = \text{length of shortest program that outputs } x$$

In practice, approximated by:
$$\hat{K}(x) = \text{len}(\text{gzip}(x))$$

### Apparent Complexity

$$C_{\text{apparent}}(x) = K(f(x))$$

where $f$ is the coarse-graining function. In practice:

$$\hat{C}(x) = \text{len}(\text{gzip}(\text{threshold}(\text{avg\_blocks}(x, g))))$$

### Sophistication (for reference)

$$\text{soph}_c(x) = \min\{K(S) : x \in S, \; K(S) + \log_2|S| \leq K(x) + c\}$$

Coarse sophistication (robust version):
$$\text{csoph}(x) = \min_c \{c + \text{soph}_c(x)\}$$

### Chernoff Bound (used in Appendix)

For sum $a_t(B)$ of independent 0/1 random variables:

$$\Pr[|a_t(B) - E[a_t(B)]| > L^2/G] < 2\exp\left(-\frac{L^2}{3G^2}\right)$$

So grain size $L \gg G\sqrt{3\ln(2n^2)} = \Theta(G\sqrt{\log n})$ ensures all grains match expectations.

### Scaling Results

$$\max(\hat{K}(x)) \sim n^2 \quad \text{(entropy, quadratic in grid size)}$$

$$\max(\hat{C}(x)) \sim n \quad \text{(complexity, linear in grid size)}$$

$$t_{\text{peak}} \sim n^2 \quad \text{(time to peak, quadratic)}$$

---

## What the Paper Gets Right

1. Clear, well-defined experiment with reproducible setup
2. Honest about failures — they report OSCR and two-part code attempts that didn't work
3. Separate analysis of interacting vs non-interacting — showed that the key result depends on interactions
4. Caught their own artifact (Section 5 vs Section 6) and fixed it
5. Analytical proof for the non-interacting case, not just numerics
6. Extensive comparison of complexity measures with clear reasoning for choices

## What the Paper Doesn't Cover

1. No proof that the interacting model *must* become complex — this is explicitly listed as an open problem
2. No comparison with light-cone complexity (mentioned as future work)
3. No systematic study of grain size selection
4. The OSCR implementation may just have been bad — the algorithm itself might work with a better 2D-aware version
5. Only tested on the specific initial condition (top-half/bottom-half). Other geometries might give different curves.
6. No connection to practical ML — this is pure complexity theory applied to a toy physical model

---

## Looking Back (Our Retrospective, Not in the Paper)

> Note: This section contains our reflections, not claims from the paper.

Ten years later, this paper remains one of the only empirical studies of how complexity evolves in a closed system. The coffee automaton is now a standard example in complexity theory lectures, though the open problem (proving the interacting model must become complex) remains unsolved as of 2024.

The choice to use gzip as a KC proxy has aged well — it's the same approach used in recent "gzip classification" papers (Jiang et al. 2023, "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors"). The principle that you can use off-the-shelf compressors to estimate Kolmogorov complexity continues to be productive.

The connection between Days 6 and 7 is direct: Aaronson proposed complextropy (Day 6 blog post, 2011), Carroll posed the coffee cup question (FQXi conference, 2011), and three years later they published this empirical study (2014). Lauren Ouellette wrote the code as an MIT undergrad.

For ML practitioners: the notion that "interestingness" peaks at intermediate stages of a mixing/optimization process has loose analogies to representation learning (where intermediate layers of deep networks capture the most "useful" features), but the paper makes no such claims, and the connection is speculative.

---

## Questions Worth Thinking About

1. The paper proves the non-interacting model stays simple. Can you prove the interacting model *must* become complex? What would such a proof look like?

2. They used gzip, which is based on LZ77+ Huffman coding. What would happen with a stronger compressor (e.g., LZMA, brotli, or a learned compressor)?

3. The grain size g is fixed experimentally. Could you define a notion of "optimal" grain size — the one that maximizes the peak complexity? Does this relate to the renormalization group?

4. What happens with different initial conditions? What if coffee and cream start in a checkerboard pattern? Or a single cream droplet in the center?

5. How does this connect to the "origin of structure" in the actual universe? Carroll (co-author and physicist) cares about this question — his book "From Eternity to Here" discusses the thermodynamic arrow of time and the specialness of the Big Bang's low-entropy state.

6. The paper shows max complexity ~ n (linear in grid side). Is there a physical system where max complexity grows faster than linearly? What would that require?
