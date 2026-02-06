# Paper Notes: The First Law of Complexodynamics

> Notes on Scott Aaronson's 2011 blog post

---

## Post Overview

**Title:** The First Law of Complexodynamics
**Author:** Scott Aaronson
**Year:** 2011
**Source:** [Blog post](https://scottaaronson.blog/?p=762)
**Context:** FQXi "Setting Time Aright" conference (cruise from Bergen to Copenhagen)

**One-sentence summary:**
*"Can we define a formal notion of 'complexity' that provably increases then decreases over time for physical systems, even as entropy monotonically increases?"*

---

## ELI5 (Explain Like I'm 5)

### The Story

You make a cup of coffee and pour in some milk. At first, the coffee and milk are separate — boring. Then as they start to mix, you get these beautiful swirling patterns — interesting! Then eventually everything is the same tan color — boring again.

The milk is mixing, so "disorder" (entropy) keeps going up the whole time. But the "interestingness" goes up and then comes back down. A kid could see this. But nobody has a math formula for "interestingness" that provably has to go up then down. Aaronson tries to find one.

> Note: This analogy is Aaronson's own — the three coffee cups example is central to the blog post.

---

## What the Post Actually Covers

Aaronson's post is structured around a specific question from Sean Carroll, proposes a formal answer (complextropy), and openly admits the main conjecture is unproven. The post is conversational and includes extensive discussion in comments that adds substantial content (particularly from Charles Bennett, Sean Carroll, and Luca Trevisan).

---

## The Core Idea (From the Post)

### The Problem

Entropy is well-defined and increases monotonically (second law of thermodynamics). "Complexity" or "interestingness" informally peaks at intermediate times. Can we:
1. Define a formal quantity that captures this?
2. Prove it must peak at intermediate times?

### Why Simple Answers Fail

**Kolmogorov complexity (KC)?** No — random strings have maximum KC but zero structure. Also, for deterministic systems, KC of the state after t steps grows only as O(log t) because you can describe it as "initial state + run t steps."

**Shannon entropy?** No — it's exactly what increases monotonically. It measures disorder, not structure.

**Mutual information?** Closer, but doesn't directly capture "interestingness" of a single state.

### The Progression of Definitions

Aaronson walks through a series of increasingly refined definitions:

1. **KC**: Fails because random = maximal KC but zero structure.
2. **Resource-bounded KC**: Require the describing program to run in polynomial time. Helps with the "log t" problem for deterministic systems, but still doesn't separate structure from randomness.
3. **Sophistication (Koppel 1988, refined by Gacs-Tromp-Vitanyi)**: Two-part code — find the smallest "model" S that contains x and within which x looks random. K(S) is the sophistication. This correctly gives low values to both simple and random strings. But for deterministic dynamics, still grows only as O(log t).
4. **Complextropy (Aaronson's proposal)**: Resource-bounded sophistication — require that BOTH the sampling algorithm (from S) and the reconstruction algorithm (from sample + extra bits to x) run in near-linear time.

---

## The Key Technical Definitions

### Sophistication

For a string x, the sophistication is (informally):

$$
\text{soph}(x) = \min \{ K(S) : x \in S, \; K(x|S) \geq \log_2|S| - c \}
$$

where:
- $S$ is a set (the "model")
- $K(S)$ is the KC of a program that enumerates $S$
- $K(x|S)$ is the KC of $x$ given $S$ — this must be near $\log_2|S|$, meaning $x$ is "random within $S$"
- $c$ is a constant

**Intuition:** Find the smallest description of a *set* that contains $x$ and within which $x$ has no further compressible patterns.

### Complextropy

Aaronson adds resource bounds to sophistication:

> "The number of bits in the shortest computer program that runs in $n \cdot \log(n)$ time, and outputs a nearly-uniform sample from a set $S$ such that:
> (i) $x \in S$, and
> (ii) any computer program that outputs $x$ in $n \cdot \log(n)$ time, given an oracle providing independent uniform samples from $S$, has at least $\log_2(|S|) - c$ bits."

**Why this matters:** Without the efficiency constraint, you can always cheat by defining S as "all possible outputs of programs of length K(x)." The time bound blocks this — at intermediate times, the set of reachable states from a physical process has genuine structure (the tendril boundaries) that can't be efficiently specified as anything simpler.

---

## The Model System

### Coffee Cup as 2D Pixel Grid

- 2D array of black (coffee) and white (milk) pixels
- Initial state: top half = black, bottom half = white (clean boundary)
- Dynamics: at each step, pick a random adjacent coffee-milk pair and swap them
- This is a discrete random diffusion process

### What Happens

| Time | State | Entropy | Complexity |
|------|-------|---------|-----------|
| t=0 | Clean half-and-half | Low | Low |
| t=mid | Fractal-like tendrils | Medium | HIGH |
| t=large | Uniform grey | High | Low |

### Why Intermediate States Are Complex

The tendril boundaries between coffee and milk regions encode information about the specific history of random swaps. Describing those boundaries takes many bits (high KC). But the state isn't random — it has structure (connected regions, conservation laws). So sophistication is also high. This is exactly the regime where complextropy should peak.

---

## The Blog Comments (Substantial Content)

### Sean Carroll (Comments #6-7)

Carroll emphasizes the importance of **coarse-graining**:
- Proposed: measure KC of a coarse-grained (blurred) version of the state
- Pointed out that complexity could "grow and crash repeatedly, not smoothly"
- Noted that uniform diffusion (like the coffee example) might keep complexity relatively low; the universe is more interesting because gravity creates structure

### Charles Bennett (Comment #110)

Detailed argument FOR logical depth as the right measure:
- Intermediate coffee states ARE logically deep because "any near-incompressible program for generating it would need to approximately simulate this physical evolution"
- Program = few bits for dynamical laws + few bits for initial condition + many bits for stochastic influences
- Final equilibrium is logically SHALLOW because "an alternative computation could short-circuit the system's actual evolution"
- Depth is in units of time/computation steps, not bits

### Aaronson's Response to Bennett (#112)

- Skeptical: "I don't see any intuitive reason why the depth should become large at intermediate times"
- Argues specifying tendril boundaries = high sophistication, but NOT high depth (given boundaries, sampling is fast)
- Can't figure out how to prove depth becomes large
- Reports Lauren Ouellette's empirical results: coarse-grained KC DOES increase then decrease

### Luca Trevisan (Comment #17)

Proposed multi-scale analysis:
- Divide space into cubes at each scale
- Compute average content per cube
- Track KC at each scale over time
- Get a 3D plot (scale x time x complexity) with a "bump" at intermediate scales and times

---

## Practical Measurement Approaches

### 1. gzip Compression

Serialize state to bytes, compress with gzip. Compressed size approximates KC.

**Strengths:** Fast, reproducible, widely understood.
**Weaknesses:** Not an optimal compressor; misses patterns that better algorithms would catch. Sensitive to serialization order.

### 2. Coarse-Grained gzip (Carroll's Measure)

Blur/downsample the image, then gzip the result.

**Strengths:** Captures macroscopic structure. Less noisy than pixel-level gzip.
**Weaknesses:** Choice of coarse-graining scale is ad hoc.

### 3. Two-Part Code (Sophistication Proxy)

Part 1: Coarse-grained description (the "model"). Part 2: Residual to reconstruct exact state.
Sophistication proxy = size of Part 1.

**Strengths:** Directly approximates sophistication. Part 1 size should show the characteristic hump.
**Weaknesses:** How you define the coarse model matters. No canonical choice.

---

## What the Post Gets Right

- Clearly identifies the gap between entropy (well-defined, monotone) and complexity (informal, non-monotone)
- Provides a concrete progression from KC to sophistication to complextropy, each fixing a specific deficiency
- Is honest about what's proven and what's conjectured
- Proposes a specific model system (coffee pixels) for empirical testing
- Connects to practical approximations (gzip, two-part codes) that are actionable

## What the Post Doesn't Cover

- No proof of the First Law conjecture (explicitly stated)
- No formal comparison to logical depth (though discussed in comments)
- No empirical results beyond preliminary mentions (those come in Day 7)
- Resource-bounded sophistication definitions are informal; formal treatment would need more precision about the time bounds and uniformity requirements
- No discussion of how complextropy relates to thermodynamic entropy production

## Looking Back (Our Retrospective, Not in the Post)

> [Our Addition: Retrospective — written 2024, not part of the original 2011 post]

The blog post turned out to be remarkably generative:
- It directly spawned the Coffee Automaton paper (Day 7, 2014) with Lauren Ouellette and Sean Carroll
- The practical approach (gzip as KC proxy) has been widely adopted in follow-up work, including the "gzip beats BERT" controversy (2023)
- The complextropy conjecture remains open as of 2024
- Bennett's logical depth argument from the comments (#110) anticipated much of the subsequent debate about computational complexity vs. descriptive complexity
- The multi-scale approach Trevisan suggested became standard in complexity measurement work

---

## Questions Worth Thinking About

1. If you had a perfect compressor (computing exact KC), would the complextropy curve look qualitatively different from the gzip curve? Or does gzip capture the essential shape?
2. Why does Aaronson reject logical depth while Bennett argues for it? Who is right? Can you construct a system where they disagree?
3. The coffee model uses random nearest-neighbor swaps. Would different dynamics (e.g., turbulent mixing, deterministic chaotic maps) produce different complextropy curves?
4. How does coarse-graining scale affect the complexity curve? Is there an "optimal" scale, and if so, what determines it?
5. [Our Addition] Can you connect complextropy to neural network training? Is there a sense in which a network's internal representations are "most complex" at some intermediate point during training?

---

**Next:** [Day 7 — The Coffee Automaton](../07_coffee_automaton/)
