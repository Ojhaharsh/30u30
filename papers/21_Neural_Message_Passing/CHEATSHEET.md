# Day 21 Cheat Sheet: Neural Message Passing for Quantum Chemistry

## The Big Idea (30 seconds)

Several graph neural network architectures are all special cases of one framework: **Message Passing Neural Networks (MPNNs)**. Each node collects messages from neighbors, updates its state, repeat for T rounds, then aggregate all node states into a graph-level prediction. The best variant uses an edge network for messages, GRU for updates, and Set2Set for readout. On QM9 (~130k molecules), this achieves chemical accuracy on 11 of 13 quantum-chemical properties. (Gilmer et al. 2017, Sections 2 and 6)

**The one picture to remember:**

```
Round t:  For each node v:
              1. COLLECT:  m_v = SUM over neighbors w of M(h_v, h_w, e_vw)
              2. UPDATE:   h_v = GRU(h_v, m_v)
          After T rounds:
              3. READOUT:  y_hat = Set2Set({h_v for all v})
```

---

## Quick Start

```bash
cd papers/21_Neural_Message_Passing
pip install -r requirements.txt

# Train MPNN on synthetic molecular data
python train_minimal.py --epochs 50 --hidden-dim 64 --n-messages 3

# Quick training check (2 epochs)
python train_minimal.py --epochs 2

# Generate visualizations
python visualization.py
```

---

## MPNN Variants at a Glance

| Component | Variant | Description | Paper Section |
|-----------|---------|-------------|---------------|
| **Message** | Edge Network | $A(e_{vw}) \cdot h_w$ — NN maps edge features to matrix | 4.1 |
| **Message** | Matrix multiply | $A_{e_{vw}} h_w$ — fixed matrix per edge type (GG-NN) | 3 |
| **Message** | Concatenate | $f([h_v, h_w, e_{vw}])$ — NN on concatenation (IN) | 3 |
| **Update** | GRU | Standard gated recurrent unit | 2 |
| **Readout** | Sum | $\sum_v h_v^T$ — simple but loses distribution info | 3 |
| **Readout** | Set2Set | LSTM-based attention over node embeddings | 4.3 |
| **Structure** | Virtual edges | Latent fully-connected graph on top of real bonds | 4.2 |
| **Structure** | Master node | Single virtual node connected to all atoms | 4.2 |
| **Architecture** | Multi-tower | Split embedding into groups, separate message passing | 4.4 |

---

## Key Hyperparameters

| Parameter | Paper Value | Typical Range | What It Does | Tips |
|-----------|------------|--------------|--------------|------|
| `hidden_dim` (d) | 200 | 64-256 | Node embedding dimension | Edge network outputs d*d matrix — memory scales quadratically |
| `n_messages` (T) | 6 | 3-8 | Message passing rounds | Must be >= graph diameter for full information flow |
| `set2set_steps` (M) | 6 | 3-12 | Set2Set processing steps | More steps = more expressive readout |
| `learning_rate` | 1e-4 | 1e-4 to 1e-3 | Adam learning rate | Paper uses 1e-4 with Adam |
| `batch_size` | 20 | 16-64 | Graphs per batch | Larger = faster but more memory |
| `edge_nn_layers` | 2 | 1-3 | Layers in edge network | More layers = more expressive but slower |

---

## Common Issues and Fixes

### Edge Network Runs Out of Memory

```python
# Problem: A(e_vw) outputs d*d matrix per edge — 40,000 floats for d=200
# Fix 1: reduce hidden_dim
hidden_dim = 64  # instead of 200

# Fix 2: use low-rank factorization (not in paper, but practical)
# Output two vectors of dim d, then outer product
class LowRankEdgeNetwork(nn.Module):
    def __init__(self, edge_dim, hidden_dim, rank=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rank * hidden_dim * 2)
        )
        self.rank = rank
        self.hidden_dim = hidden_dim
```

### Training Loss Doesn't Decrease

```python
# Problem: targets span wildly different scales
# Fix: normalize each target independently
target_means = train_targets.mean(dim=0)
target_stds = train_targets.std(dim=0)
normalized_targets = (train_targets - target_means) / (target_stds + 1e-8)

# At inference: predictions = predictions * target_stds + target_means
```

### Messages Are All Zeros

```python
# Problem: node features not initialized properly
# Fix: ensure one-hot encoding includes all atom types
atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
def atom_features(atom_type):
    """One-hot encode atom type."""
    feat = [0] * len(atom_types)
    feat[atom_types[atom_type]] = 1
    return feat
# Don't forget hydrogen atoms — QM9 includes them explicitly
```

### Graph Batching Is Wrong

```python
# Problem: when batching multiple graphs, edge indices must be offset
# Fix: track node count per graph and offset edge indices
def batch_graphs(graph_list):
    node_offset = 0
    batch_edges = []
    for g in graph_list:
        edges = g.edge_index + node_offset  # offset indices
        batch_edges.append(edges)
        node_offset += g.num_nodes
    return torch.cat(batch_edges, dim=1)
```

---

## The Math (Copy-Paste Ready)

### Message Function (Edge Network, Section 4.1)

```python
# Edge network: maps edge features to d x d transformation matrix
class EdgeNetwork(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim * hidden_dim)
        )
        self.hidden_dim = hidden_dim

    def forward(self, edge_features, neighbor_states):
        # edge_features: (n_edges, edge_dim)
        # neighbor_states: (n_edges, hidden_dim)
        A = self.nn(edge_features)  # (n_edges, d*d)
        A = A.view(-1, self.hidden_dim, self.hidden_dim)  # (n_edges, d, d)
        # Batched matrix-vector multiply
        messages = torch.bmm(A, neighbor_states.unsqueeze(-1)).squeeze(-1)
        return messages  # (n_edges, hidden_dim)
```

### GRU Update (Section 2)

```python
# Standard GRU cell — message is input, hidden state is previous node state
self.gru = nn.GRUCell(hidden_dim, hidden_dim)
# Usage:
h_v_new = self.gru(m_v, h_v)  # m_v: aggregated messages, h_v: current state
```

### Set2Set Readout (Section 4.3)

```python
class Set2Set(nn.Module):
    def __init__(self, hidden_dim, processing_steps=6):
        super().__init__()
        self.lstm = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.processing_steps = processing_steps

    def forward(self, node_embeddings, batch_indices):
        # node_embeddings: (total_nodes, hidden_dim)
        # batch_indices: (total_nodes,) — which graph each node belongs to
        n_graphs = batch_indices.max().item() + 1
        h = torch.zeros(n_graphs, node_embeddings.size(1))
        c = torch.zeros_like(h)

        for _ in range(self.processing_steps):
            # Attention over nodes in each graph
            q = h  # query: (n_graphs, d)
            q_expanded = q[batch_indices]  # (total_nodes, d)
            e = (node_embeddings * q_expanded).sum(dim=-1)  # (total_nodes,)
            # Softmax per graph (scatter_softmax)
            a = scatter_softmax(e, batch_indices)  # (total_nodes,)
            r = scatter_sum(a.unsqueeze(-1) * node_embeddings,
                           batch_indices, dim=0)  # (n_graphs, d)
            h, c = self.lstm(torch.cat([q, r], dim=-1), (h, c))

        return torch.cat([h, r], dim=-1)  # (n_graphs, 2*d)
```

---

## Debugging Checklist

- [ ] Node features are correct shape and non-zero
- [ ] Edge indices are bidirectional (if undirected graph, add both directions)
- [ ] Edge features match the edge index ordering
- [ ] Batch indices correctly assigned (no off-by-one)
- [ ] Targets normalized per-property before training
- [ ] GRU input/hidden dimensions match
- [ ] Edge network output reshaped to (n_edges, d, d) correctly
- [ ] Set2Set scatter operations respect batch boundaries
- [ ] Gradient clipping enabled (1.0 is a common default)
- [ ] Adam optimizer with lr=1e-4

---

## QM9 Properties Reference

| Target | Property | Unit | Chemical Accuracy |
|--------|----------|------|-------------------|
| mu | Dipole moment | Debye | 0.1 |
| alpha | Polarizability | Bohr^3 | 0.1 |
| HOMO | Highest occupied MO | eV | 0.043 |
| LUMO | Lowest unoccupied MO | eV | 0.043 |
| gap | HOMO-LUMO gap | eV | 0.043 |
| R2 | Electronic spatial extent | Bohr^2 | 1.2 |
| ZPVE | Zero-point vibrational energy | eV | 0.0012 |
| U0 | Internal energy at 0K | eV | 0.043 |
| U | Internal energy at 298K | eV | 0.043 |
| H | Enthalpy at 298K | eV | 0.043 |
| G | Free energy at 298K | eV | 0.043 |
| Cv | Heat capacity | cal/mol/K | 0.050 |
| Omega | Highest vibrational freq | cm^-1 | 10 |

(From Table 2 of the paper)

---

## Key Relationships

```
Message Function choices:
  simple pass:     M(h_v, h_w, e_vw) = h_w
  matrix lookup:   M(h_v, h_w, e_vw) = A_{e_vw} * h_w      (GG-NN)
  edge network:    M(h_v, h_w, e_vw) = A(e_vw) * h_w        (this paper, best)
  concatenate:     M(h_v, h_w, e_vw) = f([h_v, h_w, e_vw])  (Interaction Net)

Readout choices:
  sum:      R = SUM(h_v)            fast but loses distribution info
  gated:    R = SUM(gate(h_v) * h_v) better but still sum-based
  Set2Set:  R = LSTM-attention       best (this paper)
```

---

**Next:** [Day 22 — Deep Speech 2](../22_Deep_Speech_2/)
