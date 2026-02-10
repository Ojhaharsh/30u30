"""
Exercise 5: Property Prediction (Hard, 4/5)

Bring everything together: build a complete MPNN that takes molecular graphs
as input and predicts multiple properties. Train it and evaluate per-property
mean absolute error (MAE).

This exercise combines:
  - Graph construction (Exercise 2)
  - Edge network messages (Exercise 4)
  - Set2Set readout (Exercise 3)
  - Training loop with proper batching

Reference: Gilmer et al. 2017, Sections 2, 4, and 6 (Table 2)
"""

import sys
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for importing implementation utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
    """
    Complete MPNN for molecular property prediction.

    Combines:
    - Node encoder (linear projection)
    - T rounds of edge-network message passing with GRU update
    - Set2Set readout
    - Output MLP

    Architecture follows the best variant from Table 2.
    """
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

        # TODO: Build the complete model
        # 1. Node encoder: Linear(node_dim -> hidden_dim)
        # 2. Edge network: Sequential(Linear(edge_dim -> 64) -> ReLU -> Linear(64 -> hidden_dim * hidden_dim))
        # 3. GRU update: GRUCell(hidden_dim, hidden_dim)
        # 4. Set2Set LSTM: LSTMCell(2 * hidden_dim, hidden_dim)
        # 5. Output MLP: Sequential(Linear(2 * hidden_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim))
        self.node_encoder = None   # TODO
        self.edge_nn = None        # TODO
        self.gru = None            # TODO
        self.set2set_lstm = None   # TODO
        self.output_mlp = None     # TODO

    def forward(self, batch: BatchedGraph) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch: BatchedGraph with node_features, edge_index, edge_features,
                   batch_indices, targets

        Returns:
            (n_graphs, output_dim) predictions
        """
        # TODO: Implement the full forward pass
        #
        # Phase 1: Message Passing (T rounds)
        #   h = node_encoder(batch.node_features)
        #   for t in range(n_messages):
        #     h_w = h[source_indices]
        #     A = edge_nn(batch.edge_features).view(-1, d, d)
        #     messages = bmm(A, h_w)
        #     m = scatter_sum(messages, target_indices)
        #     h = gru(m, h)
        #
        # Phase 2: Set2Set Readout
        #   Run set2set_steps of LSTM-attention over final node states
        #   Output through MLP
        raise NotImplementedError("Implement PropertyPredictor.forward")


def train_and_evaluate(n_epochs: int = 20, hidden_dim: int = 32):
    """
    Train a PropertyPredictor on synthetic molecular data.

    This function:
    1. Generates synthetic data
    2. Creates the model
    3. Trains for n_epochs
    4. Reports final per-property MAE

    Returns:
        Tuple of (model, final_mae)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    n_targets = 3
    property_names = ['Energy', 'Polarizability', 'HOMO']

    # Generate data
    print("Generating synthetic molecular data...")
    dataset = generate_dataset(n_molecules=300, n_targets=n_targets, seed=42)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Create model
    model = PropertyPredictor(
        node_dim=len(ATOM_TYPES),
        edge_dim=len(BOND_TYPES),
        hidden_dim=hidden_dim,
        output_dim=n_targets,
        n_messages=3,
        set2set_steps=4
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"\nTraining for {n_epochs} epochs...")

    for epoch in range(1, n_epochs + 1):
        model.train()
        random.shuffle(train_data)
        batch_size = 32
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = batch_graphs_impl(train_data[i:i + batch_size])
            optimizer.zero_grad()
            predictions = model(batch)
            loss = F.l1_loss(predictions, batch.targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % max(1, n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss = {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = batch_graphs_impl(test_data[i:i + batch_size])
            preds = model(batch)
            all_preds.append(preds)
            all_targets.append(batch.targets)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Per-property MAE
    print(f"\nPer-property MAE:")
    per_prop_mae = (preds - targets).abs().mean(dim=0)
    for i, name in enumerate(property_names):
        print(f"  {name}: {per_prop_mae[i].item():.4f}")

    overall_mae = per_prop_mae.mean().item()
    print(f"  Overall: {overall_mae:.4f}")

    return model, overall_mae


# ============================================================================
# Tests
# ============================================================================


def test_property_prediction():
    """Test the PropertyPredictor model."""
    torch.manual_seed(42)
    passed = True

    print("Testing PropertyPredictor...")
    try:
        model = PropertyPredictor(
            node_dim=5, edge_dim=4, hidden_dim=16,
            output_dim=3, n_messages=2, set2set_steps=3
        )

        # Check all components are defined
        assert model.node_encoder is not None, "node_encoder must be defined"
        assert model.edge_nn is not None, "edge_nn must be defined"
        assert model.gru is not None, "gru must be defined"
        assert model.set2set_lstm is not None, "set2set_lstm must be defined"
        assert model.output_mlp is not None, "output_mlp must be defined"

        # Test on small batch
        from implementation import generate_synthetic_molecule
        graphs = [generate_synthetic_molecule(n_targets=3, seed=i) for i in range(4)]
        batch = batch_graphs_impl(graphs)
        out = model(batch)

        assert out.shape == (4, 3), f"Wrong output shape: {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        print("  PASSED")

    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test training loop (only if model works)
    if passed:
        print("\nTesting training loop (5 epochs)...")
        try:
            model, mae = train_and_evaluate(n_epochs=5, hidden_dim=16)
            assert mae < 100.0, f"MAE too large: {mae}"
            print("  Training test PASSED")
        except NotImplementedError:
            print("  NOT IMPLEMENTED")
            passed = False
        except Exception as e:
            print(f"  FAILED: {e}")
            passed = False

    if passed:
        print("\nAll property prediction tests PASSED")
    else:
        print("\nSome tests failed or are not implemented yet")


if __name__ == '__main__':
    test_property_prediction()
