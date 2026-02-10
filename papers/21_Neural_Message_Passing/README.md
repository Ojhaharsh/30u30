# Day 21: Neural Message Passing for Quantum Chemistry

> Gilmer, Schoenholz, Riley, Vinyals, Dahl (2017) — [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)

**Time:** 4-6 hours
**Prerequisites:** Day 19-20 (relational reasoning), basic graph theory (nodes, edges, adjacency), PyTorch
**Code:** PyTorch

---

## What This Paper Is Actually About

By 2017, several neural network models for learning on graphs had been proposed independently — Convolutional Networks on Graphs (Duvenaud et al. 2015), Gated Graph Neural Networks (Li et al. 2015), Interaction Networks (Battaglia et al. 2016), and others. Each had its own notation, its own message-passing scheme, and its own readout mechanism. But they all did roughly the same thing: propagate information along edges, update node states, and aggregate into a graph-level prediction.

Gilmer et al. noticed this and did something simple but important: they unified all of these models into a single framework called **Message Passing Neural Networks (MPNNs)**. The framework has exactly two phases — a message passing phase and a readout phase — and each existing model is just a specific instantiation of the message function, update function, and readout function.

They then used this framework to explore new variants and applied them to the QM9 benchmark — a dataset of ~130,000 small organic molecules with 13 quantum-chemical properties computed via Density Functional Theory (DFT). The result: predictions reaching chemical accuracy on 11 of the 13 properties (Section 6, Table 2).

The paper matters beyond chemistry. The MPNN framework became the standard way to think about graph neural networks. If you understand message passing, you understand the backbone of nearly every GNN that followed.

---

## The Core Idea

Every graph neural network does three things:

1. **Message**: Each node collects information from its neighbors.
2. **Update**: Each node updates its own state based on the collected messages.
3. **Readout**: All node states are aggregated into a single graph-level output.

```
Step 1: MESSAGE              Step 2: UPDATE              Step 3: READOUT
                                                          
  h_w ---e_vw---> m_vw        m_v + h_v ---> h_v'         {h_v'} ---> y_hat
  h_u ---e_vu---> m_vu                                     
  (neighbors send messages)   (GRU updates state)         (Set2Set aggregates)
```

The MPNN framework (Section 2) defines this precisely:

**Message passing phase** (T steps):

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$$

$$h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$$

**Readout phase**:

$$\hat{y} = R(\{h_v^T \mid v \in G\})$$

Where $M_t$ is the message function, $U_t$ is the update function, $R$ is the readout function, $h_v^t$ is the hidden state of node $v$ at step $t$, $e_{vw}$ is the edge feature between $v$ and $w$, and $N(v)$ is the set of neighbors of $v$.

Different choices of $M$, $U$, and $R$ recover different existing models. The paper's contribution is making this explicit and finding which choices work best for molecular property prediction.

---

## What the Authors Actually Showed

### The QM9 Benchmark (Section 5)

QM9 contains ~130,000 small organic molecules (H, C, N, O, F atoms, up to 9 heavy atoms). Each molecule has 13 quantum-chemical properties computed via DFT at the B3LYP/6-31G(2df,p) level. The properties include:

- Atomization energies (U0, U, H, G)
- Electronic properties (HOMO, LUMO, gap)
- Dipole moment (mu)
- Electronic spatial extent (R2)
- Zero-point vibrational energy (ZPVE)
- Heat capacity (Cv)
- Vibrational frequencies

### Results (Section 6, Table 2)

The best MPNN variant achieved chemical accuracy on 11 out of 13 targets. "Chemical accuracy" is a predefined error threshold established by the chemistry community for each property — it's the point where DFT predictions are accurate enough to be useful.

Key results from Table 2:
- Best overall: MPNN with edge network message function + GRU update + Set2Set readout
- The two targets that did NOT reach chemical accuracy: mu (dipole moment) and alpha (isotropic polarizability)
- With only graph topology (no 3D coordinates): chemical accuracy on several of the 13 targets

### Unification (Section 3)

The paper shows that existing models map to specific MPNN instantiations:

| Model | Message Function | Update Function | Readout |
|-------|-----------------|----------------|---------|
| Convolutional Networks (Duvenaud) | $M = h_w$ (neighbor state) | $U$ = neural net | Sum + neural net |
| Gated Graph NN (Li et al.) | $M = A_{e_{vw}} h_w$ (matrix per edge type) | GRU | Gated sum |
| Interaction Networks (Battaglia) | $M = f(h_v, h_w, e_{vw})$ | $U = g(h_v, m_v)$ | Sum |
| MPNN (this paper, best variant) | Edge Network: $M = A(e_{vw}) h_w$ | GRU | Set2Set |

---

## The Architecture

### 1. Message Function — Edge Network (Section 4.1)

The best-performing message function uses a neural network to process edge features:

$$M(h_v, h_w, e_{vw}) = A(e_{vw}) \cdot h_w$$

where $A(e_{vw})$ is a neural network that maps the edge feature vector $e_{vw}$ to a $d \times d$ matrix. This matrix is then applied to the neighbor's hidden state $h_w$.

This is more general than having a fixed matrix per edge type (as in GG-NNs). It handles continuous edge features (bond length, bond angle) naturally.

### 2. Update Function — GRU (Section 2)

The node state is updated using a Gated Recurrent Unit:

$$h_v^{t+1} = \text{GRU}(h_v^t, m_v^{t+1})$$

The GRU treats the aggregated message $m_v^{t+1}$ as the "input" and the previous hidden state $h_v^t$ as the "recurrent state." All weights are shared across message passing steps.

### 3. Readout Function — Set2Set (Section 4.3)

For graph-level predictions, the individual node states need to be combined into a single vector. Simple summation loses information. The paper uses **Set2Set** (Vinyals et al. 2015) — an attention-based method that iteratively queries the set of node embeddings using an LSTM:

1. Initialize a query vector $q_0$
2. At each processing step $t$: compute attention over all node embeddings, produce a weighted sum, feed into the LSTM to get the next query
3. After $M$ processing steps, output the final query-embedding pair

Set2Set is permutation-invariant (order of nodes doesn't matter) but more expressive than simple sum/mean pooling.

### 4. Virtual Graph Elements (Section 4.2)

The paper also experiments with adding "virtual edges" between all pairs of atoms (a latent fully-connected graph on top of the real molecular graph). This allows information to travel between distant atoms in fewer message passing steps. The virtual edges have their own learned features distinct from real bonds.

---

## Implementation Notes

Key decisions in `implementation.py`:

- **Graph representation**: Each molecule is a tuple of `(node_features, edge_index, edge_features, target)`. Node features are one-hot atom type + other atomic properties. Edge features encode bond type, distance, etc.
- **Edge network**: A 2-layer MLP maps edge features to a $d \times d$ matrix. The matrix is reshaped from the MLP output. This is the most parameter-heavy part.
- **GRU update**: Standard PyTorch GRU cell. The aggregated message is the input, previous node state is the hidden state.
- **Set2Set readout**: LSTM-based with $M=6$ processing steps (following the paper). The output is $2d$-dimensional (concatenation of query and weighted node embedding).
- **Batching**: Multiple graphs are batched by concatenating their node/edge lists and tracking which nodes belong to which graph. This is the standard approach in PyTorch Geometric, but we implement it from scratch here.

Things that will bite you:
- **Edge network memory**: The $d \times d$ matrix per edge is expensive. For large molecules or large $d$, this dominates memory. The paper uses $d = 200$ for QM9.
- **Message passing steps**: More steps = larger receptive field but diminishing returns and training instability. The paper uses $T = 6$ steps.
- **Node feature initialization**: Atom type is one-hot encoded. If you forget to include hydrogen atoms (which QM9 has explicitly), your graph is wrong.
- **Numerical targets**: QM9 properties span wildly different scales (e.g., dipole moment in Debye vs. atomization energy in eV). Normalize each target independently.

---

## What to Build

### Quick Start

```bash
cd papers/21_Neural_Message_Passing
pip install -r requirements.txt
python train_minimal.py --epochs 50 --hidden-dim 64 --n-messages 3
```

This trains an MPNN on synthetic molecular data and reports mean absolute error per property.

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|-----------------------------|
| 1 | Message functions (`exercise_01_message_functions.py`) | Build three message function variants and compare them |
| 2 | Graph construction (`exercise_02_graph_construction.py`) | Convert molecular data into graph tensors with proper batching |
| 3 | Readout functions (`exercise_03_readout_functions.py`) | Implement sum, mean, and Set2Set readouts and compare expressiveness |
| 4 | Edge networks (`exercise_04_edge_networks.py`) | Build the edge network that maps features to transformation matrices |
| 5 | Property prediction (`exercise_05_property_prediction.py`) | Train a full MPNN on multi-target regression and evaluate per-property MAE |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **MPNNs unify existing graph neural networks.** Convolutional Networks on Graphs, Gated Graph NNs, Interaction Networks, and others are all specific instantiations of the same message/update/readout framework. The paper makes this explicit. (Section 3)

2. **The edge network is the key architectural choice for molecular data.** Using a neural network to map edge features to transformation matrices (rather than having a fixed matrix per edge type) allows handling continuous features like bond length and angle. This was the best-performing message function. (Section 4.1)

3. **Set2Set readout outperforms simple aggregation.** For graph-level predictions, attention-based aggregation (Set2Set) captures more structure than sum or mean pooling. The improvement is consistent across targets. (Section 4.3, Table 2)

4. **Chemical accuracy is achievable from data.** On 11 of 13 QM9 targets, the MPNN matches the accuracy threshold where DFT predictions become practically useful — learned entirely from data, without hand-crafted quantum chemistry features. (Section 6, Table 2)

5. **The framework transcends chemistry.** While the experiments are on molecules, the MPNN framework applies to any domain with graph-structured data: social networks, knowledge graphs, physical simulations, program analysis. The message passing abstraction is domain-agnostic. (Section 1)

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Core MPNN: message functions, GRU update, Set2Set readout, graph batching |
| `train_minimal.py` | CLI script — train on synthetic molecular data, report per-property MAE |
| `visualization.py` | Graph visualization, message flow, training curves, per-property error bars |
| `notebook.ipynb` | Interactive walkthrough — build MPNN step by step |
| `exercises/` | 5 exercises: message functions, graph construction, readout, edge networks, full pipeline |
| `paper_notes.md` | Detailed notes on Gilmer et al. 2017 |
| `CHEATSHEET.md` | Quick reference for MPNN components and hyperparameters |

---

## Further Reading

- [Gilmer et al. 2017 — Original Paper](https://arxiv.org/abs/1704.01212) — the MPNN framework paper
- [Li et al. 2015 — Gated Graph Neural Networks](https://arxiv.org/abs/1511.05493) — the GG-NN that MPNN extends
- [Vinyals et al. 2015 — Order Matters / Set2Set](https://arxiv.org/abs/1511.06391) — the readout function used here (Day 16)
- [Battaglia et al. 2018 — Relational Inductive Biases](https://arxiv.org/abs/1806.01261) — comprehensive survey of graph networks
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) — production-grade GNN library

---

**Previous:** [Day 20 — Relational RNNs](../20_Relational_RNNs/)
**Next:** [Day 22 — Deep Speech 2](../22_Deep_Speech_2/)
