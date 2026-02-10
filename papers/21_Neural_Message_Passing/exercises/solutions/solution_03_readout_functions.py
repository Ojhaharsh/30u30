"""
Solution 3: Readout Functions

Reference: Gilmer et al. 2017, Section 4.3 and Table 2
"""

import torch
import torch.nn as nn
from typing import Optional


def scatter_sum(src, index, dim=0, dim_size=None):
    """Sum elements of src grouped by index."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(dim, index_expanded, src)
    return out


class SumReadout(nn.Module):
    """R = MLP(sum_v h_v^T)"""
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_embeddings, batch_indices):
        n_graphs = batch_indices.max().item() + 1
        # Sum node embeddings per graph
        graph_emb = scatter_sum(node_embeddings, batch_indices, dim_size=n_graphs)
        return self.mlp(graph_emb)


class Set2SetReadout(nn.Module):
    """Set2Set readout — Section 4.3, from Vinyals et al. 2015."""
    def __init__(self, hidden_dim: int, output_dim: int, processing_steps: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.processing_steps = processing_steps
        self.lstm = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_embeddings, batch_indices):
        n_graphs = batch_indices.max().item() + 1
        device = node_embeddings.device

        h = torch.zeros(n_graphs, self.hidden_dim, device=device)
        c = torch.zeros(n_graphs, self.hidden_dim, device=device)
        r = torch.zeros(n_graphs, self.hidden_dim, device=device)

        for _ in range(self.processing_steps):
            # Expand query to per-node
            q = h[batch_indices]  # (total_nodes, d)

            # Attention logits
            e = (node_embeddings * q).sum(dim=-1)  # (total_nodes,)

            # Per-graph softmax (numerically stable)
            max_vals = torch.zeros(n_graphs, device=device)
            max_vals.scatter_reduce_(
                0, batch_indices, e, reduce='amax', include_self=False
            )
            e_shifted = e - max_vals[batch_indices]
            exp_e = torch.exp(e_shifted)
            sum_exp = scatter_sum(
                exp_e.unsqueeze(-1), batch_indices, dim_size=n_graphs
            ).squeeze(-1)
            a = exp_e / (sum_exp[batch_indices] + 1e-16)  # (total_nodes,)

            # Weighted sum per graph
            r = scatter_sum(
                a.unsqueeze(-1) * node_embeddings,
                batch_indices,
                dim_size=n_graphs
            )

            # LSTM update
            lstm_input = torch.cat([h, r], dim=-1)
            h, c = self.lstm(lstm_input, (h, c))

        graph_emb = torch.cat([h, r], dim=-1)
        return self.mlp(graph_emb)


if __name__ == '__main__':
    torch.manual_seed(42)
    hidden_dim, output_dim = 8, 3
    node_embeddings = torch.randn(9, hidden_dim)
    batch_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    # Test SumReadout
    sr = SumReadout(hidden_dim, output_dim)
    out = sr(node_embeddings, batch_indices)
    assert out.shape == (2, output_dim)
    # Test permutation invariance
    perm = torch.tensor([3, 1, 4, 0, 2, 5, 6, 7, 8])
    out_perm = sr(node_embeddings[perm], batch_indices[perm])
    assert torch.allclose(out, out_perm, atol=1e-5)
    print("SumReadout: PASSED")

    # Test Set2SetReadout
    s2s = Set2SetReadout(hidden_dim, output_dim, processing_steps=3)
    out = s2s(node_embeddings, batch_indices)
    assert out.shape == (2, output_dim)
    assert not torch.allclose(out[0], out[1], atol=1e-3)
    print("Set2SetReadout: PASSED")
    print("All solutions verified.")
