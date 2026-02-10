"""
Solution 4: Edge Networks

Reference: Gilmer et al. 2017, Section 4.1
"""

import torch
import torch.nn as nn


class EdgeNetworkLayer(nn.Module):
    """Single message passing layer with edge network."""
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * hidden_dim)
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, edge_features):
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # 1. Gather sender states
        h_w = h[source_idx]  # (n_edges, d)

        # 2. Compute transformation matrices from edge features
        A = self.edge_nn(edge_features)  # (n_edges, d*d)
        A = A.view(-1, self.hidden_dim, self.hidden_dim)  # (n_edges, d, d)

        # 3. Compute messages: A @ h_w
        messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)  # (n_edges, d)

        # 4. Aggregate messages per target node (scatter sum)
        m = torch.zeros_like(h)
        m.scatter_add_(
            0,
            target_idx.unsqueeze(1).expand(-1, self.hidden_dim),
            messages
        )

        # 5. GRU update
        h_new = self.gru(m, h)
        return h_new


class MultiStepMPNN(nn.Module):
    """T rounds of edge-network message passing."""
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.mp_layer = EdgeNetworkLayer(hidden_dim, edge_dim)

    def forward(self, node_features, edge_index, edge_features):
        h = self.node_encoder(node_features)
        for _ in range(self.n_steps):
            h = self.mp_layer(h, edge_index, edge_features)
        return h


if __name__ == '__main__':
    torch.manual_seed(42)
    hidden_dim, edge_dim, node_dim = 8, 4, 5
    n_nodes, n_edges = 6, 10

    h = torch.randn(n_nodes, hidden_dim)
    node_features = torch.randn(n_nodes, node_dim)
    edge_index = torch.stack([
        torch.randint(0, n_nodes, (n_edges,)),
        torch.randint(0, n_nodes, (n_edges,))
    ])
    edge_features = torch.randn(n_edges, edge_dim)

    # Test EdgeNetworkLayer
    layer = EdgeNetworkLayer(hidden_dim, edge_dim)
    h_new = layer(h, edge_index, edge_features)
    assert h_new.shape == h.shape
    print("EdgeNetworkLayer: PASSED")

    # Test MultiStepMPNN
    mpnn = MultiStepMPNN(node_dim, edge_dim, hidden_dim, n_steps=3)
    h_final = mpnn(node_features, edge_index, edge_features)
    assert h_final.shape == (n_nodes, hidden_dim)
    print("MultiStepMPNN: PASSED")
    print("All solutions verified.")
