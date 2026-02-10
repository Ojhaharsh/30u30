"""
Exercise 4: Edge Networks (Medium, 3/5)

Build the edge network that maps continuous edge features to transformation
matrices. This is the paper's key architectural contribution (Section 4.1).

The edge network is a neural network A: R^{edge_dim} -> R^{d x d} that
takes edge features (e.g., bond length, bond angle) and produces a d x d
matrix. This matrix is then applied to the neighbor's hidden state to
compute the message.

You will:
  1. Build the edge network MLP
  2. Apply it to compute messages
  3. Handle the reshape from flat output to matrix form
  4. Integrate it into a single message passing step

Reference: Gilmer et al. 2017, Section 4.1
"""

import torch
import torch.nn as nn


class EdgeNetworkLayer(nn.Module):
    """
    A single message passing layer using the edge network.

    Combines:
    - Edge network for message computation (Section 4.1)
    - Sum aggregation of messages per node
    - GRU update of node states (Section 2)
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # TODO: Create the edge network MLP
        # Input: edge_dim -> Output: hidden_dim * hidden_dim
        # Architecture: Linear -> ReLU -> Linear
        self.edge_nn = None  # Replace with nn.Sequential(...)

        # TODO: Create the GRU cell for node state update
        # Input dim = hidden_dim (aggregated messages)
        # Hidden dim = hidden_dim (node states)
        self.gru = None  # Replace with nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, edge_features):
        """
        One round of message passing.

        Args:
            h: (n_nodes, hidden_dim) — current node hidden states
            edge_index: (2, n_edges) — [source, target] indices
            edge_features: (n_edges, edge_dim) — edge features

        Returns:
            h_new: (n_nodes, hidden_dim) — updated node hidden states

        Steps:
            1. Gather sender states: h_w = h[source_indices]
            2. Compute transformation matrices: A = edge_nn(edge_features)
            3. Reshape A from (n_edges, d*d) to (n_edges, d, d)
            4. Compute messages: msg = A @ h_w
            5. Aggregate messages per node: m_v = sum of messages to v
            6. Update: h_new = GRU(m_v, h)
        """
        # TODO: Implement the full message passing step
        #
        # Hint for step 5 (scatter sum):
        #   target_indices = edge_index[1]
        #   m = torch.zeros_like(h)
        #   m.scatter_add_(0, target_indices.unsqueeze(1).expand(-1, self.hidden_dim), messages)
        raise NotImplementedError("Implement EdgeNetworkLayer.forward")


class MultiStepMPNN(nn.Module):
    """
    Stack multiple EdgeNetworkLayer rounds (T message passing steps).

    The paper uses T=6 rounds, where each round uses the same parameters
    (weight tying across steps — same GRU, same edge network).
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps

        # TODO: Create a linear layer to project node features to hidden dim
        self.node_encoder = None  # Replace: nn.Linear(node_dim, hidden_dim)

        # TODO: Create the EdgeNetworkLayer (shared across all steps)
        self.mp_layer = None  # Replace: EdgeNetworkLayer(hidden_dim, edge_dim)

    def forward(self, node_features, edge_index, edge_features):
        """
        Run T rounds of message passing.

        Args:
            node_features: (n_nodes, node_dim)
            edge_index: (2, n_edges)
            edge_features: (n_edges, edge_dim)

        Returns:
            h: (n_nodes, hidden_dim) — final node states after T rounds
        """
        # TODO: Encode node features and run T message passing steps
        raise NotImplementedError("Implement MultiStepMPNN.forward")


# ============================================================================
# Tests
# ============================================================================


def test_edge_networks():
    """Test edge network implementations."""
    hidden_dim = 8
    edge_dim = 4
    node_dim = 5
    n_nodes = 6
    n_edges = 10

    torch.manual_seed(42)
    h = torch.randn(n_nodes, hidden_dim)
    node_features = torch.randn(n_nodes, node_dim)
    edge_index = torch.stack([
        torch.randint(0, n_nodes, (n_edges,)),
        torch.randint(0, n_nodes, (n_edges,))
    ])
    edge_features = torch.randn(n_edges, edge_dim)

    passed = True

    # Test EdgeNetworkLayer
    print("Testing EdgeNetworkLayer...")
    try:
        layer = EdgeNetworkLayer(hidden_dim, edge_dim)
        assert layer.edge_nn is not None, "self.edge_nn must be defined"
        assert layer.gru is not None, "self.gru must be defined"
        h_new = layer(h, edge_index, edge_features)
        assert h_new.shape == h.shape, f"Wrong shape: {h_new.shape}"
        assert not torch.allclose(h, h_new), "States should change after message passing"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test MultiStepMPNN
    print("Testing MultiStepMPNN...")
    try:
        mpnn = MultiStepMPNN(node_dim, edge_dim, hidden_dim, n_steps=3)
        assert mpnn.node_encoder is not None, "self.node_encoder must be defined"
        assert mpnn.mp_layer is not None, "self.mp_layer must be defined"
        h_final = mpnn(node_features, edge_index, edge_features)
        assert h_final.shape == (n_nodes, hidden_dim), f"Wrong shape: {h_final.shape}"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    if passed:
        print("\nAll edge network tests PASSED")
    else:
        print("\nSome tests failed or are not implemented yet")


if __name__ == '__main__':
    test_edge_networks()
