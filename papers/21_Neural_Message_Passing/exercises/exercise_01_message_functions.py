"""
Exercise 1: Message Functions (Easy, 2/5)

Implement three message function variants from the MPNN framework:
  1. SimpleMessage: M(h_v, h_w, e_vw) = h_w (Section 3, Duvenaud et al.)
  2. MatrixMessage: M(h_v, h_w, e_vw) = A_{e_vw} * h_w (Section 3, Li et al.)
  3. EdgeNetworkMessage: M(h_v, h_w, e_vw) = A(e_vw) * h_w (Section 4.1)

The message function determines how information flows from neighbor w to
node v along the edge (v, w). Different choices give different models.

Reference: Gilmer et al. 2017, Sections 2-4
"""

import torch
import torch.nn as nn


class SimpleMessage(nn.Module):
    """
    Simplest message: just pass the neighbor's hidden state.
    M(h_v, h_w, e_vw) = h_w

    This ignores edge features entirely. Used in Convolutional Networks
    on Graphs (Duvenaud et al. 2015).
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, h_v, h_w, edge_features):
        """
        Args:
            h_v: (n_edges, hidden_dim) — receiving node states
            h_w: (n_edges, hidden_dim) — sending node states
            edge_features: (n_edges, edge_dim) — edge features (not used here)

        Returns:
            messages: (n_edges, hidden_dim)
        """
        # TODO: Return the neighbor's hidden state as the message
        # This is literally one line.
        raise NotImplementedError("Implement SimpleMessage.forward")


class MatrixMessage(nn.Module):
    """
    Matrix-based message: one learned matrix per edge type.
    M(h_v, h_w, e_vw) = A_{e_vw} * h_w

    Edge features must be one-hot encoded (discrete edge types).
    The message is a linear transformation of h_w, with the transformation
    matrix selected by edge type.

    Used in Gated Graph Neural Networks (Li et al. 2015).
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # One transformation matrix per edge type
        # edge_dim = number of discrete edge types
        self.edge_matrices = nn.Parameter(
            torch.randn(edge_dim, hidden_dim, hidden_dim) * 0.01
        )

    def forward(self, h_v, h_w, edge_features):
        """
        Args:
            h_v: (n_edges, hidden_dim) — receiving node states
            h_w: (n_edges, hidden_dim) — sending node states
            edge_features: (n_edges, edge_dim) — one-hot edge type

        Returns:
            messages: (n_edges, hidden_dim)

        Hint: Use torch.einsum to combine edge features with matrices.
              edge_features selects/blends which matrix to apply.
        """
        # TODO: Implement matrix-based message passing
        # Step 1: Compute the per-edge transformation matrix A
        #         A = sum_k(edge_features[k] * edge_matrices[k]) for each edge
        #         Use: A = torch.einsum('ek,khd->ehd', edge_features, self.edge_matrices)
        # Step 2: Apply A to h_w using batched matrix multiply
        #         messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)
        raise NotImplementedError("Implement MatrixMessage.forward")


class EdgeNetworkMessage(nn.Module):
    """
    Edge network message: a neural network maps edge features to a
    transformation matrix.
    M(h_v, h_w, e_vw) = A(e_vw) * h_w

    This is the paper's key architectural contribution (Section 4.1).
    Unlike MatrixMessage, this handles continuous edge features naturally.
    """
    def __init__(self, hidden_dim: int, edge_dim: int, edge_hidden: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # TODO: Create a 2-layer MLP that maps edge features to d*d values
        # The MLP should be:
        #   Linear(edge_dim -> edge_hidden) -> ReLU -> Linear(edge_hidden -> hidden_dim * hidden_dim)
        #
        # Hint: Use nn.Sequential
        self.nn = None  # Replace with your implementation

    def forward(self, h_v, h_w, edge_features):
        """
        Args:
            h_v: (n_edges, hidden_dim) — receiving node states
            h_w: (n_edges, hidden_dim) — sending node states
            edge_features: (n_edges, edge_dim) — continuous edge features

        Returns:
            messages: (n_edges, hidden_dim)

        Steps:
            1. Pass edge_features through the MLP to get (n_edges, d*d) output
            2. Reshape to (n_edges, d, d)
            3. Matrix-vector multiply with h_w
        """
        # TODO: Implement edge network message passing
        raise NotImplementedError("Implement EdgeNetworkMessage.forward")


# ============================================================================
# Tests
# ============================================================================


def test_message_functions():
    """Test all three message function implementations."""
    hidden_dim = 8
    edge_dim = 4
    n_edges = 10

    # Random test data
    torch.manual_seed(42)
    h_v = torch.randn(n_edges, hidden_dim)
    h_w = torch.randn(n_edges, hidden_dim)
    # One-hot edge features
    edge_indices = torch.randint(0, edge_dim, (n_edges,))
    edge_features = torch.zeros(n_edges, edge_dim)
    edge_features.scatter_(1, edge_indices.unsqueeze(1), 1.0)

    passed = True

    # Test SimpleMessage
    print("Testing SimpleMessage...")
    try:
        simple = SimpleMessage(hidden_dim, edge_dim)
        out = simple(h_v, h_w, edge_features)
        assert out.shape == (n_edges, hidden_dim), f"Wrong shape: {out.shape}"
        assert torch.allclose(out, h_w), "SimpleMessage should just return h_w"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test MatrixMessage
    print("Testing MatrixMessage...")
    try:
        matrix = MatrixMessage(hidden_dim, edge_dim)
        out = matrix(h_v, h_w, edge_features)
        assert out.shape == (n_edges, hidden_dim), f"Wrong shape: {out.shape}"
        # Check that output depends on edge features (different edges give different messages)
        assert not torch.allclose(out[0], out[1], atol=1e-4), \
            "Different edges should generally give different messages"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test EdgeNetworkMessage
    print("Testing EdgeNetworkMessage...")
    try:
        edge_net = EdgeNetworkMessage(hidden_dim, edge_dim)
        assert edge_net.nn is not None, "self.nn must be defined"
        out = edge_net(h_v, h_w, edge_features)
        assert out.shape == (n_edges, hidden_dim), f"Wrong shape: {out.shape}"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    if passed:
        print("\nAll message function tests PASSED")
    else:
        print("\nSome tests failed or are not implemented yet")


if __name__ == '__main__':
    test_message_functions()
