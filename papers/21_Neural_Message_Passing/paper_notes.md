# Paper Notes: Neural Message Passing for Quantum Chemistry

> Notes on Gilmer, Schoenholz, Riley, Vinyals, Dahl (2017)

---

## Paper Overview

**Title:** Neural Message Passing for Quantum Chemistry
**Authors:** Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
**Year:** 2017
**Published at:** ICML 2017
**Source:** [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
**Affiliation:** Google Brain

**One-sentence summary:**
*"Several existing graph neural network architectures are all special cases of a single Message Passing Neural Network framework, and a particularly effective variant achieves chemical accuracy on 11 of 13 molecular property targets."*

---

## ELI5 (Explain Like I'm 5)

### The Story

Imagine you have a Lego molecule — each brick is an atom, and the connections between bricks are bonds. You want to predict something about the whole molecule (say, how much energy it stores) just by looking at its shape.

Here is how you do it: each atom (brick) looks at its neighbors, collects notes about what they look like and what kind of bond connects them. Then each atom updates its own description based on those notes. After a few rounds of this — atoms passing notes to their neighbors, updating themselves — every atom has a description that reflects not just itself but its local chemical environment. Then you add up all the descriptions to get a single number for the whole molecule.

That is message passing. The paper's contribution is showing that several different groups of researchers had independently invented variations of this same idea, and then finding the best combination of choices for molecular prediction.

> Note: This analogy is ours. The paper does not use a Lego analogy — it describes the framework in mathematical terms (Section 2).

---

## What the Paper Actually Covers

The paper has a clear structure:

1. **The MPNN Framework (Section 2)** — formal definition of message passing, update, and readout phases
2. **Existing Models as MPNNs (Section 3)** — shows that Convolutional Networks on Graphs, GG-NNs, Interaction Networks, Molecular Graph Convolutions, and Deep Tensor NNs are all special cases
3. **New MPNN Variants (Section 4)** — proposes edge networks, virtual graph elements, Set2Set readout, and multi-tower architectures
4. **The QM9 Dataset (Section 5)** — describes the benchmark: ~130,000 molecules, 13 DFT-computed properties
5. **Results (Section 6)** — state-of-the-art on all 13 targets, chemical accuracy on 11/13

The paper does NOT propose entirely new concepts — the individual components (GRU for updates, attention for readout, neural networks for messages) all existed. The contribution is the unification and the systematic search for the best combination.

---

## The Core Idea (From the Paper)

### The Framework (Section 2)

An MPNN operates on an undirected graph $G$ with node features $x_v$ and edge features $e_{vw}$. It runs in two phases:

**Phase 1 — Message Passing** (T rounds):

At each round $t$:

1. For each node $v$, compute messages from all neighbors:

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$$

2. Update each node's hidden state:

$$h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$$

where $h_v^0 = x_v$ (initial hidden state is the node features).

**Phase 2 — Readout**:

Aggregate all final node states into a graph-level prediction:

$$\hat{y} = R(\{h_v^T \mid v \in G\})$$

$M_t$, $U_t$, and $R$ are the three design choices. Different choices give different models.

### Why This Matters

Before this paper, each graph NN paper defined its own framework from scratch. This made it hard to compare them or understand what actually mattered. By reducing everything to three functions ($M$, $U$, $R$), the paper enables systematic comparison and mix-and-match architecture search.

---

## How Existing Models Fit (Section 3)

The paper demonstrates that five existing architectures are MPNN special cases:

### Convolutional Networks on Graphs (Duvenaud et al. 2015)

- Message: $M(h_v, h_w, e_{vw}) = h_w$ (just pass the neighbor's state)
- Update: $U = \sigma(H_{deg(v)}^t \cdot m_v^{t+1})$ (degree-specific weight matrix)
- Readout: $R = \sum_v f(h_v^T)$ (sum of per-node softmax)

### Gated Graph Neural Networks (Li et al. 2015)

- Message: $M(h_v, h_w, e_{vw}) = A_{e_{vw}} h_w$ (one matrix per edge type)
- Update: GRU
- Readout: Gated sum $R = \sum_v \sigma(i(h_v^T, x_v)) \odot j(h_v^T)$

### Interaction Networks (Battaglia et al. 2016)

- Message: $M(h_v, h_w, e_{vw}) = f([h_v, h_w, e_{vw}])$ (neural net on concatenation)
- Update: $U = g(h_v, m_v)$
- Readout: $R = f(\sum_v h_v^T)$

### Molecular Graph Convolutions (Kearnes et al. 2016)

- Introduces edge hidden states $e_{vw}^t$ that also get updated each round
- Message includes both node and edge states

### Deep Tensor Neural Networks (Schutt et al. 2017)

- Uses distance-based edge features (continuous, not categorical)
- Applies a "cfconv" layer specialized for spatial data

---

## The New Variants (Section 4)

### Edge Network (Section 4.1)

Instead of having one weight matrix per edge type (which only works for discrete edge types like single/double/triple bonds), use a neural network $A$ that maps the edge feature vector to a transformation matrix:

$$M(h_v, h_w, e_{vw}) = A(e_{vw}) \cdot h_w$$

where $A: \mathbb{R}^{d_e} \to \mathbb{R}^{d \times d}$ is a neural network.

This handles continuous edge features (distances, angles) naturally. The output of $A$ is a $d \times d$ matrix, so if $d = 200$, that is 40,000 outputs from the edge network per edge — parameter-heavy but effective.

### Virtual Graph Elements (Section 4.2)

Add "virtual edges" between every pair of atoms, even those not bonded. Virtual edges have their own learnable features, separate from real bond features. This creates a latent fully-connected graph on top of the real molecular graph.

The benefit: information travels between any two atoms in a single message passing step, rather than needing $T$ steps to traverse $T$ bonds.

The paper also tries a "master node" — a virtual node connected to all real atoms — as a simpler alternative.

### Set2Set Readout (Section 4.3)

Simple sum/mean pooling loses information about the distribution of node states. Set2Set (Vinyals et al. 2015, Day 16) iteratively attends to the node embeddings:

1. Initialize query $q_0$ via LSTM
2. For $M$ processing steps:
   - Compute attention weights: $a_i = \text{softmax}(q_t^T h_i)$
   - Weighted sum: $r_t = \sum_i a_i h_i$
   - LSTM update: $q_{t+1} = \text{LSTM}(q_t, r_t)$
3. Output: concatenation $[q_M, r_M]$

This is permutation-invariant and more expressive than sum pooling.

### Multiple Towers (Section 4.4)

Split the node embedding into $k$ groups and run separate message passing on each group. This is analogous to multi-head attention — each "tower" can learn different aspects of the molecular structure.

---

## The QM9 Experiments

### Dataset (Section 5)

- ~130,000 molecules of H, C, N, O, F atoms
- Up to 9 heavy (non-hydrogen) atoms per molecule
- 13 quantum-chemical properties per molecule, computed with DFT at B3LYP/6-31G(2df,p) level
- Properties span different physical domains: energies (eV), spatial extent (Bohr^2), frequencies (cm^-1), dipole moment (Debye)

Two experimental settings:
1. **With spatial information**: full 3D coordinates of atoms as input (distances as continuous edge features)
2. **Without spatial information**: only atom types and bond types (graph topology only)

### Results (Section 6, Table 2)

With spatial information:
- Chemical accuracy achieved on 11 of 13 targets
- Failed on: mu (dipole moment) and alpha (isotropic polarizability)
- Best architecture: edge network + GRU + Set2Set

Without spatial information (graph topology only):
- Chemical accuracy on several of the 13 targets
- This is notable: the model implicitly infers something about 3D geometry from the graph structure alone

### What "Chemical Accuracy" Means

For each of the 13 properties, the chemistry community has defined an error threshold below which predictions are practically useful. These thresholds vary by property — for some it is 0.043 eV, for others it is 0.1 Debye. Table 2 reports the MAE for each target alongside the chemical accuracy threshold.

---

## The Key Technical Definitions

### Message Aggregation

The sum over neighbors:

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$$

Sum is the default. The paper does not experiment with max or mean aggregation — those explorations came later (Xu et al. 2018, Hamilton et al. 2017).

### GRU Update

$$h_v^{t+1} = \text{GRU}(h_v^t, m_v^{t+1})$$

The GRU gates control how much of the old state to keep vs. how much to update from the message. Weights are tied across all time steps $t$ — the same GRU is applied at every message passing round.

### Set2Set

Iterative attention over node embeddings. The output is a $2d$-dimensional vector that captures the distribution of node states, not just their sum. The paper uses $M = 6$ processing steps for Set2Set.

---

## What the Paper Gets Right

- Identifies the key insight that several independent lines of work are doing the same thing
- Clean, minimal framework — three functions, two phases, done
- Systematic experimental comparison with clear ablations
- Honest about the chemical accuracy metric (reports both successes and failures)
- The Section 3 taxonomy of existing models is a genuine reference contribution

## What the Paper Doesn't Cover

- No theoretical analysis of expressiveness (what functions can/can't MPNNs compute? — answered later by Xu et al. 2018 "How Powerful are GNNs?")
- No experiments on large molecules or proteins (QM9 has at most 9 heavy atoms)
- No comparison of aggregation operators (sum vs. max vs. mean) — sum is assumed throughout
- No attention-based message functions (Graph Attention Networks came later, Velickovic et al. 2018)
- Edge state updates are mentioned (Molecular Graph Convolutions) but not systematically explored
- The paper acknowledges that Set2Set is best but does not deeply analyze WHY (Section 6)

## Looking Back (Our Retrospective, Not in the Paper)

> [Our Addition: Retrospective — written 2024, not part of the original 2017 paper]

The MPNN framework became the de facto way to describe graph neural networks. Virtually every subsequent GNN paper defines its model in terms of the MPNN functions:

- **Graph Attention Networks (GATs, 2018)**: attention-weighted messages — a specific $M_t$ choice
- **GIN (Xu et al. 2018)**: proved that sum aggregation is maximally expressive for distinguishing graphs, theoretically justifying the paper's default choice
- **SchNet, DimeNet, PaiNN (2017-2021)**: specialized MPNNs for 3D molecular data, extending the edge network idea to spherical harmonics and directional messages
- **PyTorch Geometric**: the standard library for GNNs, implements the `MessagePassing` base class directly from this framework

The paper's key limitation — small molecules only — has been addressed by subsequent work on protein-ligand interaction, materials science, and drug discovery, all built on the MPNN foundation.

---

## Questions Worth Thinking About

1. The paper uses sum aggregation for messages. Xu et al. (2018) later proved that sum is maximally expressive. But for practical molecular prediction, does this actually matter? Would max or mean aggregation give similar results on QM9?

2. Virtual edges create a fully connected graph. At what point does adding virtual connections become equivalent to just running a standard neural network on a flattened feature vector? When does graph structure stop helping?

3. The edge network maps features to $d \times d$ matrices. For $d = 200$, that is 40,000 parameters per edge. Is there a more parameter-efficient way to condition messages on edge features?

4. Chemical accuracy was not reached for dipole moment and polarizability. These are global properties that depend on the spatial distribution of electrons. Is this a fundamental limitation of message passing on local neighborhoods?

5. [Our Addition] How does the number of message passing steps ($T$) interact with the graph diameter? For QM9 (diameter typically 3-5), $T = 6$ is enough for every atom to see every other atom. But for proteins (diameter 20+), would you need $T = 20$? Or do virtual edges solve this?

---

**Previous:** [Day 20 — Relational RNNs](../20_Relational_RNNs/)
**Next:** [Day 22 — Deep Speech 2](../22_Deep_Speech_2/)
