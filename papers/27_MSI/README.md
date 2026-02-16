# Day 27: Machine Super Intelligence

> Shane Legg (2008) — [Machine Super Intelligence](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf)

**Time:** 4-6 hours
**Prerequisites:** Day 26 (Kolmogorov Complexity), Basic Reinforcement Learning
**Code:** Python (NumPy)

---

## What This Paper Is Actually About

Shane Legg's doctoral thesis provides a formal, mathematical definition of intelligence. Instead of relying on human-centric benchmarks (like the Turing Test) or specific tasks (like Chess or Go), Legg proposes a "universal intelligence" measure denoted as $\Upsilon$.

The core argument is that intelligence is an agent's ability to achieve its goals across a wide range of environments. By weighting these environments based on their complexity (using Kolmogorov complexity), we can derive a single number that represents an agent's "general" intelligence. This work serves as the theoretical foundation for much of modern AGI research and AI safety.

---

## What Legg Actually Showed

In his analysis, Legg unified dozens of existing definitions of intelligence and derived a formal measure that is non-anthropocentric.

1.  **Universal Intelligence $\Upsilon$**: He showed that an agent's intelligence can be calculated as a weighted average of its expected reward across all possible computable environments.
2.  **AIXI Optimality**: He formally analyzed AIXI (Hutter's universal agent), proving that it achieves the maximum possible universal intelligence score, though it is non-computable.
3.  **Complexity Dependence**: He demonstrated that a "smart" agent must be able to recognize patterns in simple environments quickly while still having the capacity to handle complex ones (Table 4.1).

---

## The Core Idea

The central mechanism is **Algorithmic Information Theory** applied to **Reinforcement Learning**.

### The Agent-Environment Loop

Shane Legg uses a formal model of an interaction sequence where an agent and environment exchange signals.

```
          ┌──────────────────────────────────────────┐
          │               ENVIRONMENT (μ)            │
          │  Program that generates observations     │
          │  and rewards based on past actions.      │
          └─────┬──────────────────────────────▲─────┘
                │   Observation (o_t)          │   
                │   Reward (r_t)               │   Action (a_t)
                │                              │
          ┌─────▼──────────────────────────────┴─────┐
          │                 AGENT (π)                │
          │  Policy that chooses the next action     │
          │  based on the history of interaction.    │
          └──────────────────────────────────────────┘
```

The intelligence of agent $\pi$ is defined as its expected performance across the space of all computable reward-summarizing environments:

$$\Upsilon(\pi) = \sum_{\mu \in E} 2^{-K(\mu)} V_{\mu}^\pi$$

### Step-by-Step: The Intelligence Score

1.  **Sample the Space ($E$):** Consider every possible computer program that could be an environment.
2.  **Weighted Sum:** We don't just average the performance. We weight it by $2^{-K(\mu)}$, where $K(\mu)$ is the Kolmogorov complexity (Day 26).
3.  **Occam's Razor:** This weighting means that "simpler" environments (like a GridWorld) carry much more weight than "complex" or "noisy" ones (like a chaotic system).
4.  **Universal Metric:** The result is a single number $\Upsilon$ that describes how "generally" capable the agent is at finding patterns and maximizing reward.

---

## The Architecture

Following Legg's thesis, we decompose the interaction into three major pillars:

1.  **The Agent ($\pi$):** A mapping from histories (past actions, observations, rewards) to a probability distribution over actions.
2.  **The Environment ($\mu$):** A computable function. Legg notes that almost everything we care about (physics, games, logic) is computable.
3.  **Universal Distribution:** The weight $2^{-K(\mu)}$ serves as a universal "prior." An intelligent agent acts as if simple environments are more likely to be the true one.

---

## Implementation Notes

Our implementation provides a simulation framework to estimate $\Upsilon$ using a curated environment suite of varying complexity.

-   **The Coffee Automaton Example:** To illustrate complexity, we include environments inspired by Aaronson's Coffee Automaton (Day 7)—ranging from simple mixing to complex chaotic states.
-   **Predictive Boxing:** Instead of a generic RL agent, we implement a **PredictiveAgent** that attempts to compress the environment's history to predict future rewards—a direct nod to the "Intelligence = Compression" principle.
-   **Numerical Stability:** We handle the $2^{-K}$ weights in log-space to avoid underflow when dealing with high-complexity tasks.

---

## What to Build

### Quick Start

```bash
# Run the intelligence evaluation suite
python train_minimal.py

# Generate the Intelligence vs. Complexity visualization
python visualization.py
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Implement the $\Upsilon$ formula | Understanding how complexity weights affect global scores. |
| 2 | Code a "Simpleton" vs. "Pattern-Seeker" | Seeing why recognizing patterns is the key to universal intelligence. |
| 3 | Environment Complexity Estimation | Applying Day 26 concepts to measure how "hard" a task is. |
| 4 | The AIXI-Lite Agent | Implementing a basic predictive agent that uses Solomonoff-style induction. |
| 5 | Scoring Normalization | Learning how to compare an agent's performance across wildly different domains. |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1.  **[Definition]** — Intelligence is general ability, not specialized skill (Section 4.1).
2.  **[Compression]** — The ability to seek out and exploit patterns in environments is the core of intelligence (Section 2.4).
3.  **[Theoretical Limit]** — AIXI represents the upper bound of intelligence but is practically non-computable due to its search over all programs (Section 5.3).

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | The Universal Intelligence Measure framework and environment suite. |
| `visualization.py` | Plots agent performance across the complexity spectrum. |
| `train_minimal.py` | Benchmarks three different agent types (Random, Heuristic, RL). |
| `paper_notes.md` | Deep dive into the math and philosophical implications. |
| `CHEATSHEET.md` | Quick reference for formulas and agent types. |

---

## Further Reading

-   [Machine Super Intelligence](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf) - Original Thesis (Shane Legg, 2008)
-   [Universal Intelligence: A Definition of Machine Intelligence](https://arxiv.org/abs/0712.3344) - Journal paper (Legg & Hutter, 2007)
-   [Philosophical implications of AIXI](https://hutter1.net/ai/aixigv.htm) - Marcus Hutter's overview

---

**Next:** [Day 28 — CS231n: CNNs for Visual Recognition](../28_cs231n/)
