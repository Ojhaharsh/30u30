"""
Exercise 2: Graph Construction (Easy, 2/5)

Build molecular graph representations from raw data. A molecule is
represented as a graph where:
  - Nodes = atoms (features: one-hot atom type)
  - Edges = bonds (features: one-hot bond type)
  - Edges are bidirectional (undirected graph)

You need to:
  1. Convert atom lists and bond lists into proper tensor format
  2. Ensure edges are bidirectional
  3. Batch multiple graphs into a single tensor set with proper index offsets

Reference: Gilmer et al. 2017, Section 5 (QM9 dataset description)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


# Atom and bond type vocabularies (from QM9, Section 5)
ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
BOND_TYPES = {'single': 0, 'double': 1, 'triple': 2, 'aromatic': 3}


def atoms_to_features(atom_list: List[str]) -> torch.Tensor:
    """
    Convert a list of atom type strings to a one-hot feature tensor.

    Args:
        atom_list: e.g., ['C', 'C', 'O', 'H', 'H', 'H', 'H']

    Returns:
        (n_atoms, 5) tensor — one-hot encoded atom types

    Example:
        atoms_to_features(['C', 'O']) should return:
        tensor([[0, 1, 0, 0, 0],   # C is index 1
                [0, 0, 0, 1, 0]])   # O is index 3
    """
    # TODO: Create a (n_atoms, len(ATOM_TYPES)) tensor of zeros
    # Then set the appropriate index to 1.0 for each atom
    raise NotImplementedError("Implement atoms_to_features")


def bonds_to_edge_data(
    bond_list: List[Tuple[int, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of bonds to edge_index and edge_features tensors.

    Each bond is a tuple (atom_i, atom_j, bond_type).
    Since the molecular graph is undirected, each bond must be represented
    as TWO directed edges: (i -> j) and (j -> i).

    Args:
        bond_list: e.g., [(0, 1, 'single'), (1, 2, 'double')]

    Returns:
        edge_index: (2, n_edges) tensor of [source, target] indices
        edge_features: (n_edges, 4) tensor of one-hot bond types

    Example:
        bonds_to_edge_data([(0, 1, 'single')]) should return:
        edge_index = tensor([[0, 1],   # source: 0->1 and 1->0
                             [1, 0]])   # target
        edge_features = tensor([[1, 0, 0, 0],   # single bond
                                [1, 0, 0, 0]])   # same type both ways
    """
    # TODO: Build the edge_index and edge_features tensors
    # Remember: each bond becomes TWO directed edges
    #
    # Steps:
    # 1. For each (i, j, bond_type) in bond_list:
    #    - Add edges (i, j) and (j, i) to the source/target lists
    #    - Add the bond type one-hot encoding twice (once per direction)
    # 2. Stack into tensors
    raise NotImplementedError("Implement bonds_to_edge_data")


def batch_graphs(
    node_features_list: List[torch.Tensor],
    edge_index_list: List[torch.Tensor],
    edge_features_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch multiple graphs into concatenated tensors with proper index offsets.

    When batching, edge indices must be offset by the cumulative node count
    so that each graph's node indices remain unique. A batch_indices tensor
    tracks which graph each node belongs to.

    Args:
        node_features_list: List of (n_nodes_i, node_dim) tensors
        edge_index_list: List of (2, n_edges_i) tensors
        edge_features_list: List of (n_edges_i, edge_dim) tensors

    Returns:
        batched_node_features: (total_nodes, node_dim)
        batched_edge_index: (2, total_edges) — with offsets applied
        batched_edge_features: (total_edges, edge_dim)
        batch_indices: (total_nodes,) — graph index for each node
    """
    # TODO: Concatenate all graphs with proper edge index offsets
    #
    # Key insight: When graph 0 has 5 nodes and graph 1 has 3 nodes,
    # the edge indices in graph 1 need to be shifted by +5 so they
    # refer to the correct nodes in the concatenated tensor.
    #
    # Steps:
    # 1. Track cumulative node_offset = 0
    # 2. For each graph i:
    #    - Add node_features to list
    #    - Add edge_index + node_offset to list
    #    - Add edge_features to list
    #    - Create batch_indices = [i, i, ..., i] for n_nodes_i nodes
    #    - Update node_offset += n_nodes_i
    # 3. Concatenate all lists
    raise NotImplementedError("Implement batch_graphs")


# ============================================================================
# Tests
# ============================================================================


def test_graph_construction():
    """Test graph construction functions."""
    passed = True

    # Test atoms_to_features
    print("Testing atoms_to_features...")
    try:
        features = atoms_to_features(['C', 'O', 'H'])
        assert features.shape == (3, 5), f"Wrong shape: {features.shape}"
        assert features[0, 1] == 1.0, "C should be at index 1"
        assert features[1, 3] == 1.0, "O should be at index 3"
        assert features[2, 0] == 1.0, "H should be at index 0"
        assert features.sum() == 3.0, "Each atom should have exactly one 1.0"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test bonds_to_edge_data
    print("Testing bonds_to_edge_data...")
    try:
        edge_index, edge_features = bonds_to_edge_data([
            (0, 1, 'single'),
            (1, 2, 'double')
        ])
        assert edge_index.shape == (2, 4), f"Wrong edge_index shape: {edge_index.shape}"
        assert edge_features.shape == (4, 4), f"Wrong edge_features shape: {edge_features.shape}"
        # Check bidirectionality
        sources = set(edge_index[0].tolist())
        targets = set(edge_index[1].tolist())
        assert sources == {0, 1, 1, 2} or len(sources) == 3, "Edges should cover all bond endpoints"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test batch_graphs
    print("Testing batch_graphs...")
    try:
        nf1 = torch.randn(3, 5)  # Graph 0: 3 nodes
        ei1 = torch.tensor([[0, 1], [1, 0]])  # 1 bidirectional edge
        ef1 = torch.randn(2, 4)

        nf2 = torch.randn(4, 5)  # Graph 1: 4 nodes
        ei2 = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 directed edges
        ef2 = torch.randn(3, 4)

        b_nf, b_ei, b_ef, b_idx = batch_graphs(
            [nf1, nf2], [ei1, ei2], [ef1, ef2]
        )

        assert b_nf.shape == (7, 5), f"Wrong batched node shape: {b_nf.shape}"
        assert b_ei.shape == (2, 5), f"Wrong batched edge shape: {b_ei.shape}"
        assert b_ef.shape == (5, 4), f"Wrong batched feature shape: {b_ef.shape}"
        assert b_idx.shape == (7,), f"Wrong batch indices shape: {b_idx.shape}"

        # Graph 1's edges should be offset by 3
        assert b_ei[0, 2].item() >= 3, "Graph 1 edge indices should be offset by 3"
        assert b_idx[:3].tolist() == [0, 0, 0], "First 3 nodes should be graph 0"
        assert b_idx[3:].tolist() == [1, 1, 1, 1], "Last 4 nodes should be graph 1"

        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    if passed:
        print("\nAll graph construction tests PASSED")
    else:
        print("\nSome tests failed or are not implemented yet")


if __name__ == '__main__':
    test_graph_construction()
