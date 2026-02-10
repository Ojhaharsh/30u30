"""
visualization.py - MPNN Visualization Tools

Visualizations for understanding Message Passing Neural Networks:
1. Molecular graph structure (nodes = atoms, edges = bonds)
2. Message passing flow (how information propagates)
3. Training curves (loss and MAE over epochs)
4. Per-property error analysis (MAE bar chart)
5. Edge attention heatmap (which edges carry the most information)

Based on Gilmer et al. 2017, "Neural Message Passing for Quantum Chemistry".
"""

import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from implementation import (
    MolecularGraph,
    MPNN,
    ATOM_TYPES,
    generate_synthetic_molecule,
    batch_graphs,
)


# Color scheme for atom types (consistent with standard molecular visualization)
ATOM_COLORS = {
    'H': '#FFFFFF',   # white
    'C': '#808080',   # grey
    'N': '#3050F8',   # blue
    'O': '#FF0D0D',   # red
    'F': '#90E050',   # green
}

ATOM_SIZES = {
    'H': 200,
    'C': 400,
    'N': 350,
    'O': 350,
    'F': 300,
}


def plot_molecule(
    graph: MolecularGraph,
    title: str = 'Molecular Graph',
    save_path: Optional[str] = None,
    show_labels: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize a molecular graph with colored atoms.

    Args:
        graph: MolecularGraph to visualize.
        title: Plot title.
        save_path: Path to save the figure. If None, returns the figure.
        show_labels: Whether to label nodes with atom types.

    Returns:
        matplotlib Figure if save_path is None.
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        print("matplotlib and networkx required for visualization")
        return None

    # Build networkx graph
    G = nx.Graph()
    n_nodes = graph.num_nodes

    # Determine atom types from one-hot features
    atom_indices = graph.node_features.argmax(dim=1).numpy()
    atom_labels = [ATOM_TYPES[i] for i in atom_indices]

    for i in range(n_nodes):
        G.add_node(i, atom_type=atom_labels[i])

    # Add edges (deduplicate since our graph is bidirectional)
    edge_set = set()
    edges = graph.edge_index.numpy()
    for j in range(edges.shape[1]):
        src, tgt = edges[0, j], edges[1, j]
        if src < tgt:
            edge_set.add((src, tgt))
    G.add_edges_from(edge_set)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Node colors and sizes
    colors = [ATOM_COLORS.get(atom_labels[i], '#808080') for i in range(n_nodes)]
    sizes = [ATOM_SIZES.get(atom_labels[i], 300) for i in range(n_nodes)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#CCCCCC', width=2.0)

    # Draw nodes with atom-type coloring
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors, node_size=sizes,
        edgecolors='black', linewidths=1.5
    )

    if show_labels:
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={i: atom_labels[i] for i in range(n_nodes)},
            font_size=10, font_weight='bold'
        )

    # Legend
    legend_handles = []
    for atom_type in ATOM_TYPES:
        if atom_type in atom_labels:
            patch = mpatches.Patch(
                color=ATOM_COLORS[atom_type],
                label=atom_type,
                edgecolor='black',
                linewidth=0.5
            )
            legend_handles.append(patch)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    ax.set_title(title, fontsize=13)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        return None
    return fig


def plot_message_passing_steps(
    graph: MolecularGraph,
    n_steps: int = 3,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Visualize how messages propagate through the graph over T rounds.

    Shows the "receptive field" of a selected node at each message passing
    step — after T steps, a node has received information from all nodes
    within T hops (Section 2).

    Args:
        graph: MolecularGraph to visualize.
        n_steps: Number of message passing steps.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure if save_path is None.
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        print("matplotlib and networkx required for visualization")
        return None

    # Build networkx graph
    G = nx.Graph()
    n_nodes = graph.num_nodes
    atom_indices = graph.node_features.argmax(dim=1).numpy()
    atom_labels = [ATOM_TYPES[i] for i in atom_indices]

    for i in range(n_nodes):
        G.add_node(i)

    edges = graph.edge_index.numpy()
    edge_set = set()
    for j in range(edges.shape[1]):
        src, tgt = edges[0, j], edges[1, j]
        if src < tgt:
            edge_set.add((src, tgt))
    G.add_edges_from(edge_set)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Choose a central node (the one with highest degree)
    center = max(range(n_nodes), key=lambda n: G.degree(n))

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(4 * (n_steps + 1), 4))
    if n_steps == 0:
        axes = [axes]

    for step in range(n_steps + 1):
        ax = axes[step]

        # Find all nodes within 'step' hops of center
        reachable = set()
        current_layer = {center}
        for _ in range(step):
            next_layer = set()
            for node in current_layer:
                next_layer.update(G.neighbors(node))
            current_layer = next_layer - reachable
            reachable.update(current_layer)
        reachable.add(center)

        # Color nodes: center=red, reachable=orange, unreachable=grey
        colors = []
        for i in range(n_nodes):
            if i == center:
                colors.append('#FF4444')
            elif i in reachable:
                colors.append('#FFB347')
            else:
                colors.append('#E0E0E0')

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#CCCCCC', width=1.5)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=colors, node_size=300,
            edgecolors='black', linewidths=1.0
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={i: atom_labels[i] for i in range(n_nodes)},
            font_size=8
        )

        ax.set_title(f'T={step}: {len(reachable)} nodes reached', fontsize=11)
        ax.axis('off')

    fig.suptitle(
        f'Message Passing Receptive Field (Section 2)\n'
        f'Red = target node, Orange = reachable after T steps',
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        return None
    return fig


def plot_training_curves(
    train_losses: List[float],
    test_metrics: List[Dict[str, float]],
    title: str = 'MPNN Training',
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot training loss and test MAE over epochs.

    Args:
        train_losses: List of training losses per epoch.
        test_metrics: List of dicts with 'mae' and 'mse' keys.
        title: Plot title.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure if save_path is None.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    # Training loss
    axes[0].plot(epochs, train_losses, color='#2196F3', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss (MAE)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Test MAE
    test_maes = [m['mae'] for m in test_metrics]
    axes[1].plot(epochs, test_maes, color='#FF5722', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test MAE')
    axes[1].set_title('Test MAE')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        return None
    return fig


def plot_per_property_mae(
    property_names: List[str],
    maes: List[float],
    chemical_accuracy: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Bar chart of MAE per predicted property.

    If chemical_accuracy thresholds are provided (from Table 2), they are
    shown as horizontal dashed lines for comparison.

    Args:
        property_names: Names of the properties.
        maes: Mean absolute error for each property.
        chemical_accuracy: Chemical accuracy thresholds (from Table 2).
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure if save_path is None.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = np.arange(len(property_names))
    bars = ax.bar(x, maes, color='#4CAF50', alpha=0.8, label='MPNN MAE')

    if chemical_accuracy is not None:
        for i, threshold in enumerate(chemical_accuracy):
            if threshold is not None:
                ax.hlines(
                    threshold, i - 0.4, i + 0.4,
                    colors='red', linestyles='dashed', linewidth=2
                )
        # Single legend entry for chemical accuracy line
        ax.hlines([], 0, 0, colors='red', linestyles='dashed',
                  label='Chemical Accuracy (Table 2)')

    ax.set_xlabel('Property')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Per-Property MAE vs Chemical Accuracy Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(property_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        return None
    return fig


def plot_message_function_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Compare training curves for different message function variants.

    Shows how simple, matrix, and edge_network message functions
    converge (Section 3-4 comparison).

    Args:
        results: Dict mapping message_type -> {'losses': [...], 'metrics': [...]}.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure if save_path is None.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return None

    colors = {
        'simple': '#2196F3',
        'matrix': '#FF9800',
        'edge_network': '#4CAF50'
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for msg_type, result in results.items():
        color = colors.get(msg_type, '#808080')
        epochs = range(1, len(result['losses']) + 1)

        axes[0].plot(epochs, result['losses'], color=color,
                     label=msg_type, linewidth=1.5)
        test_maes = [m['mae'] for m in result['metrics']]
        axes[1].plot(epochs, test_maes, color=color,
                     label=msg_type, linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss by Message Function')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test MAE')
    axes[1].set_title('Test MAE by Message Function')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Message Function Comparison (Sections 3-4)', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        return None
    return fig


# ============================================================================
# MAIN: Generate all standard visualizations
# ============================================================================


if __name__ == '__main__':
    output_dir = '.'
    print("Generating MPNN visualizations...")
    print("=" * 50)

    # 1. Molecule visualization
    print("\n1. Molecule graph visualization")
    mol = generate_synthetic_molecule(min_atoms=8, max_atoms=12, seed=42)
    plot_molecule(
        mol,
        title=f'Synthetic Molecule ({mol.num_nodes} atoms, {mol.num_edges // 2} bonds)',
        save_path=os.path.join(output_dir, 'molecule_graph.png')
    )

    # 2. Message passing receptive field
    print("\n2. Message passing receptive field")
    plot_message_passing_steps(
        mol, n_steps=3,
        save_path=os.path.join(output_dir, 'message_passing_steps.png')
    )

    # 3. Per-property MAE example (using QM9 property names from Table 2)
    print("\n3. Per-property MAE (synthetic example)")
    qm9_props = ['mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'R2', 'ZPVE',
                 'U0', 'U', 'H', 'G', 'Cv']
    # Synthetic MAE values (not real results)
    synthetic_maes = [0.15, 0.12, 0.035, 0.038, 0.05, 1.0, 0.001,
                      0.04, 0.04, 0.04, 0.04, 0.045]
    # Chemical accuracy thresholds from Table 2
    chem_acc = [0.1, 0.1, 0.043, 0.043, 0.043, 1.2, 0.0012,
                0.043, 0.043, 0.043, 0.043, 0.050]

    plot_per_property_mae(
        qm9_props, synthetic_maes,
        chemical_accuracy=chem_acc,
        save_path=os.path.join(output_dir, 'per_property_mae.png')
    )

    print("\nAll visualizations generated.")
