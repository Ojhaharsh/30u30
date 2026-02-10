"""
Solution 1: Message Functions

Reference: Gilmer et al. 2017, Sections 2-4
"""

import torch
import torch.nn as nn


class SimpleMessage(nn.Module):
    """M(h_v, h_w, e_vw) = h_w — Section 3, Duvenaud et al."""
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, h_v, h_w, edge_features):
        return h_w


class MatrixMessage(nn.Module):
    """M(h_v, h_w, e_vw) = A_{e_vw} * h_w — Section 3, Li et al."""
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_matrices = nn.Parameter(
            torch.randn(edge_dim, hidden_dim, hidden_dim) * 0.01
        )

    def forward(self, h_v, h_w, edge_features):
        # Weighted combination of matrices based on edge types
        A = torch.einsum('ek,khd->ehd', edge_features, self.edge_matrices)
        # Apply to neighbor state
        messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)
        return messages


class EdgeNetworkMessage(nn.Module):
    """M(h_v, h_w, e_vw) = A(e_vw) * h_w — Section 4.1"""
    def __init__(self, hidden_dim: int, edge_dim: int, edge_hidden: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nn = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, hidden_dim * hidden_dim)
        )

    def forward(self, h_v, h_w, edge_features):
        # Map edge features to d x d matrix
        A = self.nn(edge_features)  # (n_edges, d*d)
        A = A.view(-1, self.hidden_dim, self.hidden_dim)  # (n_edges, d, d)
        # Matrix-vector multiply
        messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)
        return messages


if __name__ == '__main__':
    # Quick validation
    torch.manual_seed(42)
    hidden_dim, edge_dim, n_edges = 8, 4, 10
    h_v = torch.randn(n_edges, hidden_dim)
    h_w = torch.randn(n_edges, hidden_dim)
    edge_features = torch.zeros(n_edges, edge_dim)
    edge_features.scatter_(1, torch.randint(0, edge_dim, (n_edges, 1)), 1.0)

    for name, cls in [('Simple', SimpleMessage), ('Matrix', MatrixMessage),
                      ('EdgeNetwork', EdgeNetworkMessage)]:
        msg_fn = cls(hidden_dim, edge_dim)
        out = msg_fn(h_v, h_w, edge_features)
        assert out.shape == (n_edges, hidden_dim)
        print(f"  {name}: PASSED (shape {out.shape})")
    print("All solutions verified.")
