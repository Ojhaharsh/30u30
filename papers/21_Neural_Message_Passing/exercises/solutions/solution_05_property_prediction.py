"""
Solution 5: Property Prediction

Reference: Gilmer et al. 2017, Sections 2, 4, and 6
"""

import sys
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    MolecularGraph,
    BatchedGraph,
    batch_graphs as batch_graphs_impl,
    generate_dataset,
    ATOM_TYPES,
    BOND_TYPES,
    scatter_sum,
    scatter_softmax,
)


class PropertyPredictor(nn.Module):
    """Complete MPNN for molecular property prediction."""
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 3,
        n_messages: int = 3,
        set2set_steps: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_messages = n_messages
        self.set2set_steps = set2set_steps

        # Node encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # Edge network (Section 4.1)
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * hidden_dim)
        )

        # GRU update (Section 2)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Set2Set readout (Section 4.3)
        self.set2set_lstm = nn.LSTMCell(2 * hidden_dim, hidden_dim)

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, batch: BatchedGraph) -> torch.Tensor:
        # Phase 1: Message Passing
        h = self.node_encoder(batch.node_features)

        source_idx = batch.edge_index[0]
        target_idx = batch.edge_index[1]

        for t in range(self.n_messages):
            # Gather neighbor states
            h_w = h[source_idx]

            # Edge network: map features to transformation matrix
            A = self.edge_nn(batch.edge_features)
            A = A.view(-1, self.hidden_dim, self.hidden_dim)

            # Compute messages
            messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)

            # Aggregate per node
            m = scatter_sum(messages, target_idx, dim_size=h.size(0))

            # GRU update
            h = self.gru(m, h)

        # Phase 2: Set2Set Readout
        n_graphs = batch.batch_indices.max().item() + 1
        device = h.device

        s2s_h = torch.zeros(n_graphs, self.hidden_dim, device=device)
        s2s_c = torch.zeros(n_graphs, self.hidden_dim, device=device)
        r = torch.zeros(n_graphs, self.hidden_dim, device=device)

        for _ in range(self.set2set_steps):
            q = s2s_h[batch.batch_indices]
            e = (h * q).sum(dim=-1)

            # Per-graph softmax
            a = scatter_softmax(e, batch.batch_indices, dim_size=n_graphs)

            r = scatter_sum(
                a.unsqueeze(-1) * h,
                batch.batch_indices,
                dim_size=n_graphs
            )

            lstm_input = torch.cat([s2s_h, r], dim=-1)
            s2s_h, s2s_c = self.set2set_lstm(lstm_input, (s2s_h, s2s_c))

        graph_emb = torch.cat([s2s_h, r], dim=-1)
        return self.output_mlp(graph_emb)


def train_and_evaluate(n_epochs: int = 20, hidden_dim: int = 32):
    """Train and evaluate on synthetic data."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    n_targets = 3
    property_names = ['Energy', 'Polarizability', 'HOMO']

    dataset = generate_dataset(n_molecules=300, n_targets=n_targets, seed=42)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    model = PropertyPredictor(
        node_dim=len(ATOM_TYPES), edge_dim=len(BOND_TYPES),
        hidden_dim=hidden_dim, output_dim=n_targets,
        n_messages=3, set2set_steps=4
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32

    for epoch in range(1, n_epochs + 1):
        model.train()
        random.shuffle(train_data)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = batch_graphs_impl(train_data[i:i + batch_size])
            optimizer.zero_grad()
            preds = model(batch)
            loss = F.l1_loss(preds, batch.targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % max(1, n_epochs // 5) == 0 or epoch == 1:
            print(f"Epoch {epoch}: loss = {epoch_loss / n_batches:.4f}")

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = batch_graphs_impl(test_data[i:i + batch_size])
            all_preds.append(model(batch))
            all_targets.append(batch.targets)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    per_prop_mae = (preds - targets).abs().mean(dim=0)

    print("\nPer-property MAE:")
    for i, name in enumerate(property_names):
        print(f"  {name}: {per_prop_mae[i].item():.4f}")

    overall = per_prop_mae.mean().item()
    print(f"  Overall: {overall:.4f}")
    return model, overall


if __name__ == '__main__':
    train_and_evaluate(n_epochs=20, hidden_dim=32)
