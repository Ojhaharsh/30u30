"""
visualization.py - Research-Grade Visualization Suite for Relation Networks

This module provides high-density visualizations to test the limits of 
Relational Reasoning. 

Features:
1. Generalization Curves: Plotting accuracy vs. number of objects (N).
2. Aggregator Comparison: Visualizing the 'Counting' bias (Sum vs. Mean).
3. Relational Heatmaps: Deep-dive into internal g_theta magnitudes.
4. Failure Case Mapping: Spatial analysis of where the model breaks.

Usage:
    python visualization.py --compare-aggregators
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from implementation import RelationNetwork
from train_minimal import RelationalDataset, run_evaluation

# ============================================================================
# RESEARCH PLOTS
# ============================================================================

def plot_generalization_curve(model, device, train_n: int, max_n: int = 20):
    """
    Plots model accuracy as the number of objects (N) increases.
    Tests if the RN logic generalizes to unseen set sizes.
    """
    n_values = range(2, max_n + 1)
    accuracies = []
    
    print(f"Generating Generalization Curve (Train N={train_n})...")
    
    for n in n_values:
        test_dataset = RelationalDataset(mode='count', num_samples=500, num_objects=n)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        # Note: Counting task requires f_phi to have num_objects+1 outputs.
        # This visualization assumes the model was built for the largest N.
        acc = run_evaluation(model, test_loader, device)
        accuracies.append(acc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, accuracies, 'bo-', linewidth=2, label='RN Performance')
    plt.axvline(train_n, color='r', linestyle='--', label=f'Training N={train_n}')
    
    plt.xlabel("Number of Objects (N)")
    plt.ylabel("Accuracy (Counting Task)")
    plt.title("Set Size Generalization: Accuracy vs. Object Count", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    
    save_path = "viz_generalization.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Output] Saved generalization curve to {save_path}")


def plot_aggregator_comparison(results_dict: dict):
    """
    Visualizes the performance of Sum vs Mean aggregators on a counting task.
    """
    labels = list(results_dict.keys())
    accs = [results_dict[l] for l in labels]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    plt.ylabel("Accuracy")
    plt.title("Relational Inductive Bias: Sum vs. Mean Aggregation", fontsize=14)
    plt.ylim(0, 1.0)
    
    # Add accuracy labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.savefig("viz_aggregator_comp.png", dpi=150)
    plt.close()
    print("  [Output] Saved aggregator comparison to viz_aggregator_comp.png")


def plot_failure_heatmap(points, model, device):
    """
    Visualizes g_theta 'attention' for a failure case. 
    Helps identify if the model is ignoring certain objects.
    """
    model.eval()
    with torch.no_grad():
        points_t = torch.tensor(points).float().unsqueeze(0).to(device)
        pairs = model.generate_pairs(points_t)
        g_out = model.g_theta(pairs.view(-1, pairs.shape[-1]))
        rel_mag = torch.norm(g_out, dim=-1).view(len(points), len(points)).cpu().numpy()
        
    plt.figure(figsize=(8, 7))
    plt.imshow(rel_mag, cmap='viridis')
    plt.colorbar(label='Relation Magnitude')
    plt.title("Internal Relation Weights (Analogy to Section 4.1 Analysis)")
    plt.xlabel("Object Index i")
    plt.ylabel("Object Index j")
    plt.savefig("viz_internal_heatmap.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Smoke Test: Dummy Visualization Suite
    device = torch.device('cpu')
    dummy_model = RelationNetwork(object_dim=2, output_dim=21, aggregator='sum')
    
    # 1. Generalization
    plot_generalization_curve(dummy_model, device, train_n=10, max_n=20)
    
    # 2. Aggregator Comp (Mock data)
    mock_results = {'sum': 0.92, 'mean': 0.45, 'max': 0.61}
    plot_aggregator_comparison(mock_results)
    
    print("\nVisualization Smoke Test Complete.")
