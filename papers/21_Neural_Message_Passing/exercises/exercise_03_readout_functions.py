"""
Exercise 3: Readout Functions (Medium, 3/5)

Implement two readout functions that aggregate node embeddings into a
single graph-level representation:

  1. SumReadout: R = MLP(sum_v h_v^T) — simple permutation-invariant baseline
  2. Set2SetReadout: LSTM-attention based aggregation (Section 4.3)

The readout function must be permutation-invariant: reordering the nodes
should not change the output. Sum achieves this trivially. Set2Set achieves
it through attention weights that depend on content, not position.

Reference: Gilmer et al. 2017, Section 4.3 and Table 2
"""

import torch
import torch.nn as nn
from typing import Optional


def scatter_sum(src, index, dim=0, dim_size=None):
    """Sum elements of src grouped by index. Helper function."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(dim, index_expanded, src)
    return out


class SumReadout(nn.Module):
    """
    Sum readout: aggregate by summing all node embeddings per graph,
    then pass through an MLP for prediction.

    R = MLP(sum_v h_v^T)
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: Create a 2-layer MLP:
        #   Linear(hidden_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim)
        self.mlp = None  # Replace with your implementation

    def forward(self, node_embeddings, batch_indices):
        """
        Args:
            node_embeddings: (total_nodes, hidden_dim)
            batch_indices: (total_nodes,) — which graph each node belongs to

        Returns:
            (n_graphs, output_dim) — one prediction per graph
        """
        # TODO: Implement sum readout
        # Step 1: Sum node embeddings per graph using scatter_sum
        # Step 2: Pass through MLP
        raise NotImplementedError("Implement SumReadout.forward")


class Set2SetReadout(nn.Module):
    """
    Set2Set readout (Section 4.3): attention-based aggregation.

    Uses an LSTM to iteratively attend to different subsets of nodes.
    After processing_steps iterations, outputs [query, weighted_sum].

    The paper uses 6 processing steps and reports that Set2Set consistently
    outperforms sum pooling across all QM9 targets (Table 2).
    """
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        processing_steps: int = 6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.processing_steps = processing_steps

        # TODO: Create the LSTM cell and output MLP
        # LSTM input dim = 2 * hidden_dim (concatenation of query and weighted sum)
        # LSTM hidden dim = hidden_dim
        # Output MLP: Linear(2 * hidden_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim)
        self.lstm = None  # Replace: nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.mlp = None   # Replace with your implementation

    def forward(self, node_embeddings, batch_indices):
        """
        Args:
            node_embeddings: (total_nodes, hidden_dim)
            batch_indices: (total_nodes,) — which graph each node belongs to

        Returns:
            (n_graphs, output_dim) — one prediction per graph

        Algorithm (from Vinyals et al. 2015, used in Section 4.3):
            1. Initialize h, c = zeros(n_graphs, hidden_dim)
            2. For each processing step:
               a. q = h (current query)
               b. e_i = dot(q[batch[i]], node_embeddings[i]) for all nodes
               c. a_i = softmax(e_i) per graph (attention weights)
               d. r = sum(a_i * node_embeddings[i]) per graph (weighted sum)
               e. h, c = LSTM([q, r], (h, c))
            3. Return MLP(cat(h, r))
        """
        # TODO: Implement Set2Set readout
        # This is the hardest part of the exercises. Follow the algorithm above.
        #
        # Hint for per-graph softmax:
        #   max_vals = scatter_max(e, batch_indices)  # or compute manually
        #   e_shifted = e - max_vals[batch_indices]   # numerical stability
        #   exp_e = torch.exp(e_shifted)
        #   sum_exp = scatter_sum(exp_e, batch_indices)
        #   a = exp_e / sum_exp[batch_indices]
        raise NotImplementedError("Implement Set2SetReadout.forward")


# ============================================================================
# Tests
# ============================================================================


def test_readout_functions():
    """Test readout function implementations."""
    hidden_dim = 8
    output_dim = 3
    torch.manual_seed(42)

    # Create fake batched graph data: 2 graphs, 5+4=9 nodes
    node_embeddings = torch.randn(9, hidden_dim)
    batch_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    passed = True

    # Test SumReadout
    print("Testing SumReadout...")
    try:
        readout = SumReadout(hidden_dim, output_dim)
        assert readout.mlp is not None, "self.mlp must be defined"
        out = readout(node_embeddings, batch_indices)
        assert out.shape == (2, output_dim), f"Wrong shape: {out.shape}"

        # Check permutation invariance: shuffle nodes within graph 0
        perm = torch.tensor([3, 1, 4, 0, 2, 5, 6, 7, 8])
        out_perm = readout(node_embeddings[perm], batch_indices[perm])
        assert torch.allclose(out, out_perm, atol=1e-5), \
            "SumReadout should be permutation-invariant"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test Set2SetReadout
    print("Testing Set2SetReadout...")
    try:
        readout = Set2SetReadout(hidden_dim, output_dim, processing_steps=3)
        assert readout.lstm is not None, "self.lstm must be defined"
        assert readout.mlp is not None, "self.mlp must be defined"
        out = readout(node_embeddings, batch_indices)
        assert out.shape == (2, output_dim), f"Wrong shape: {out.shape}"

        # Check that different graphs give different outputs
        assert not torch.allclose(out[0], out[1], atol=1e-3), \
            "Different graphs should produce different predictions"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    if passed:
        print("\nAll readout function tests PASSED")
    else:
        print("\nSome tests failed or are not implemented yet")


if __name__ == '__main__':
    test_readout_functions()
