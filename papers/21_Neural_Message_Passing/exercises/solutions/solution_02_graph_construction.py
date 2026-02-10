"""
Solution 2: Graph Construction

Reference: Gilmer et al. 2017, Section 5 (QM9 dataset)
"""

import torch
from typing import List, Tuple

ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
BOND_TYPES = {'single': 0, 'double': 1, 'triple': 2, 'aromatic': 3}


def atoms_to_features(atom_list: List[str]) -> torch.Tensor:
    """Convert atom type strings to one-hot feature tensor."""
    n_atoms = len(atom_list)
    n_types = len(ATOM_TYPES)
    features = torch.zeros(n_atoms, n_types)
    for i, atom in enumerate(atom_list):
        features[i, ATOM_TYPES[atom]] = 1.0
    return features


def bonds_to_edge_data(
    bond_list: List[Tuple[int, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert bond list to bidirectional edge_index and edge_features."""
    sources, targets = [], []
    feature_list = []
    n_types = len(BOND_TYPES)

    for atom_i, atom_j, bond_type in bond_list:
        bond_idx = BOND_TYPES[bond_type]
        one_hot = [0.0] * n_types
        one_hot[bond_idx] = 1.0

        # Add both directions (undirected graph)
        sources.extend([atom_i, atom_j])
        targets.extend([atom_j, atom_i])
        feature_list.extend([one_hot, one_hot])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_features = torch.tensor(feature_list, dtype=torch.float32)
    return edge_index, edge_features


def batch_graphs(
    node_features_list: List[torch.Tensor],
    edge_index_list: List[torch.Tensor],
    edge_features_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch multiple graphs with proper edge index offsets."""
    batched_nodes = []
    batched_edges = []
    batched_edge_feats = []
    batch_indices = []

    node_offset = 0
    for i, (nf, ei, ef) in enumerate(zip(
        node_features_list, edge_index_list, edge_features_list
    )):
        n_nodes = nf.size(0)
        batched_nodes.append(nf)
        batched_edges.append(ei + node_offset)  # offset edge indices
        batched_edge_feats.append(ef)
        batch_indices.append(torch.full((n_nodes,), i, dtype=torch.long))
        node_offset += n_nodes

    return (
        torch.cat(batched_nodes, dim=0),
        torch.cat(batched_edges, dim=1),
        torch.cat(batched_edge_feats, dim=0),
        torch.cat(batch_indices, dim=0)
    )


if __name__ == '__main__':
    # Test atoms_to_features
    features = atoms_to_features(['C', 'O', 'H'])
    assert features.shape == (3, 5)
    assert features[0, 1] == 1.0  # C at index 1
    print("atoms_to_features: PASSED")

    # Test bonds_to_edge_data
    ei, ef = bonds_to_edge_data([(0, 1, 'single'), (1, 2, 'double')])
    assert ei.shape == (2, 4)  # 2 bonds * 2 directions
    assert ef.shape == (4, 4)
    print("bonds_to_edge_data: PASSED")

    # Test batch_graphs
    nf1 = torch.randn(3, 5)
    ei1 = torch.tensor([[0, 1], [1, 0]])
    ef1 = torch.randn(2, 4)
    nf2 = torch.randn(4, 5)
    ei2 = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ef2 = torch.randn(3, 4)

    b_nf, b_ei, b_ef, b_idx = batch_graphs([nf1, nf2], [ei1, ei2], [ef1, ef2])
    assert b_nf.shape == (7, 5)
    assert b_ei.shape == (2, 5)
    assert b_idx[:3].tolist() == [0, 0, 0]
    assert b_idx[3:].tolist() == [1, 1, 1, 1]
    assert b_ei[0, 2].item() >= 3  # graph 1 edges offset by 3
    print("batch_graphs: PASSED")
    print("All solutions verified.")
