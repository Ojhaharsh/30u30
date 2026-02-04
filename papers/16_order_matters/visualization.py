"""
🎨 Visualization Utilities for Pointer Networks

This file helps you SEE what the model learns!

Visualizations included:
- Attention heatmaps: Where does the model look at each step?
- Tour plots: Visual TSP solutions
- Convex hull diagrams: Geometric understanding
- Sorting progress: Watch sorting in action

Think of these as X-ray glasses for your neural network - you can finally
see WHAT it's paying attention to and WHY it makes each decision!

Author: 30u30 AI Papers Project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Optional

# ==============================================================================
# 🎨 ATTENTION HEATMAP
# ==============================================================================
# This is the classic visualization from the paper - shows which input elements
# get attention at each output step. It's like a "gaze tracker" for the model!

def plot_attention_heatmap(
    attention_weights: List[torch.Tensor],
    input_labels: List[str],
    output_labels: List[str],
    title: str = "🔍 Pointer Attention Heatmap",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Create a beautiful attention heatmap showing where the model looks.
    
    This answers: "For each output element, which input did the model focus on?"
    
    Example output:
    
                Input Elements →
                5.2   1.8   9.3   2.1
        Step 1 [0.1] [0.7] [0.1] [0.1]  ← Focuses on 1.8 (smallest)
        Step 2 [0.1] [0.0] [0.1] [0.8]  ← Focuses on 2.1 (next smallest)
        Step 3 [0.8] [0.0] [0.1] [0.0]  ← Focuses on 5.2
        Step 4 [0.0] [0.0] [1.0] [0.0]  ← Focuses on 9.3 (largest)
    
    Args:
        attention_weights: List of [seq_len] tensors (one per output step)
        input_labels: Labels for input elements (x-axis)
        output_labels: Labels for output steps (y-axis)
        title: Plot title
        save_path: Where to save figure (optional)
        figsize: Figure size
    """
    # Convert list of tensors to 2D array
    attention_matrix = torch.stack(attention_weights).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (0=ignore, 1=focus)', fontsize=12, rotation=270, labelpad=20)
    
    # Labels
    ax.set_xticks(range(len(input_labels)))
    ax.set_xticklabels(input_labels, fontsize=11)
    ax.set_yticks(range(len(output_labels)))
    ax.set_yticklabels(output_labels, fontsize=11)
    
    ax.set_xlabel('📥 Input Elements', fontsize=13, fontweight='bold')
    ax.set_ylabel('📤 Output Steps', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Grid for readability
    ax.set_xticks(np.arange(-0.5, len(input_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(output_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add attention values as text
    for i in range(len(output_labels)):
        for j in range(len(input_labels)):
            value = attention_matrix[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved attention heatmap to {save_path}")
    
    return fig


# ==============================================================================
# 📈 SORTING VISUALIZATION
# ==============================================================================
# Watch the sorting process step by step - like a flip-book animation!

def visualize_sorting(
    input_values: torch.Tensor,
    pointers: torch.Tensor,
    attention_weights: List[torch.Tensor],
    save_path: Optional[str] = None
):
    """
    Show how the model sorts numbers step by step.
    
    Creates a multi-panel figure:
    - Left: Original unsorted values
    - Middle: Attention heatmap
    - Right: Sorted output
    
    Args:
        input_values: [seq_len] - Original numbers
        pointers: [seq_len] - Indices in sorted order
        attention_weights: List of attention distributions
        save_path: Where to save
    """
    input_values = input_values.cpu().numpy().squeeze()
    pointers = pointers.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 5))
    
    # Panel 1: Input values
    ax1 = plt.subplot(1, 3, 1)
    x_pos = np.arange(len(input_values))
    bars1 = ax1.bar(x_pos, input_values, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Input Position', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('🔀 Unsorted Input', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, input_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Attention heatmap
    ax2 = plt.subplot(1, 3, 2)
    attention_matrix = torch.stack(attention_weights).cpu().numpy()
    im = ax2.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Input Position', fontsize=12)
    ax2.set_ylabel('Output Step', fontsize=12)
    ax2.set_title('🔍 Attention Weights', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Panel 3: Sorted output
    ax3 = plt.subplot(1, 3, 3)
    sorted_values = input_values[pointers]
    bars3 = ax3.bar(x_pos, sorted_values, color='seagreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Output Position', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('✅ Sorted Output', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, sorted_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved sorting visualization to {save_path}")
    
    return fig


# ==============================================================================
# 🗺️ TSP TOUR VISUALIZATION
# ==============================================================================
# See the traveling salesman tour - which route did the model choose?

def visualize_tsp_tour(
    cities: torch.Tensor,
    tour: torch.Tensor,
    title: str = "📦 TSP Tour",
    tour_length: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10)
):
    """
    Visualize a TSP tour showing cities and the path.
    
    This is like a GPS route visualization - shows which cities to visit
    in which order, and draws the path between them.
    
    Args:
        cities: [num_cities, 2] - (x, y) coordinates
        tour: [num_cities] - Visit order
        title: Plot title
        tour_length: Total distance (optional, will be shown)
        save_path: Where to save
        figsize: Figure size
    """
    cities = cities.cpu().numpy()
    tour = tour.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cities as points
    ax.scatter(cities[:, 0], cities[:, 1], s=300, c='steelblue', 
              alpha=0.7, edgecolors='black', linewidths=2, zorder=3, label='Cities')
    
    # Label each city
    for i, (x, y) in enumerate(cities):
        ax.text(x, y, str(i), fontsize=12, ha='center', va='center',
               color='white', fontweight='bold', zorder=4)
    
    # Draw tour path
    tour_cities = cities[tour]
    # Close the tour by returning to start
    tour_cities_closed = np.vstack([tour_cities, tour_cities[0]])
    
    ax.plot(tour_cities_closed[:, 0], tour_cities_closed[:, 1],
           'r-', linewidth=2.5, alpha=0.6, zorder=2, label='Tour Path')
    
    # Mark start city with a star
    ax.scatter(cities[tour[0], 0], cities[tour[0], 1],
              s=600, c='gold', marker='*', edgecolors='black',
              linewidths=2, zorder=5, label='Start City')
    
    # Add arrows to show direction
    for i in range(len(tour)):
        start = tour_cities[i]
        end = tour_cities[(i + 1) % len(tour)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ax.arrow(start[0] + dx*0.2, start[1] + dy*0.2,
                dx*0.6, dy*0.6,
                head_width=0.02, head_length=0.02,
                fc='darkred', ec='darkred', alpha=0.5, zorder=2)
    
    # Title with tour length if provided
    if tour_length is not None:
        title = f"{title}\nTotal Distance: {tour_length:.4f}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X Coordinate', fontsize=13)
    ax.set_ylabel('Y Coordinate', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved TSP tour to {save_path}")
    
    return fig


# ==============================================================================
# 🎒 CONVEX HULL VISUALIZATION
# ==============================================================================
# Show which points are on the boundary - geometry in action!

def visualize_convex_hull(
    points: torch.Tensor,
    hull_indices: torch.Tensor,
    title: str = "🎒 Convex Hull",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10)
):
    """
    Visualize a convex hull - the "rubber band" around points.
    
    Shows:
    - All points (blue dots)
    - Hull points (red stars)
    - Hull boundary (red polygon)
    
    Args:
        points: [num_points, 2] - (x, y) coordinates
        hull_indices: [hull_size] - Indices of boundary points
        title: Plot title
        save_path: Where to save
        figsize: Figure size
    """
    points = points.cpu().numpy()
    hull_indices = hull_indices.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], s=200, c='steelblue',
              alpha=0.6, edgecolors='black', linewidths=1.5,
              zorder=2, label='All Points')
    
    # Highlight hull points
    hull_points = points[hull_indices]
    ax.scatter(hull_points[:, 0], hull_points[:, 1], s=400, c='red',
              marker='*', edgecolors='black', linewidths=2,
              zorder=4, label='Hull Points')
    
    # Draw hull polygon
    hull_points_closed = np.vstack([hull_points, hull_points[0]])
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1],
           'r-', linewidth=3, alpha=0.6, zorder=3, label='Hull Boundary')
    
    # Label hull points with their order
    for i, (x, y) in enumerate(hull_points):
        ax.text(x, y + 0.03, str(i), fontsize=10, ha='center',
               va='bottom', color='darkred', fontweight='bold', zorder=5)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X Coordinate', fontsize=13)
    ax.set_ylabel('Y Coordinate', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved convex hull to {save_path}")
    
    return fig


# ==============================================================================
# 🎬 DEMO FUNCTION
# ==============================================================================

if __name__ == "__main__":
    print("🎨 Pointer Networks Visualization Demo")
    print("=" * 60)
    
    # Create output directory
    Path("visualizations").mkdir(exist_ok=True)
    
    # Demo 1: Sorting
    print("\n📊 Demo 1: Sorting Visualization")
    input_vals = torch.tensor([5.2, 1.8, 9.3, 2.1])
    pointers = torch.tensor([1, 3, 0, 2])  # Sorted order indices
    attention_weights = [
        torch.tensor([0.1, 0.7, 0.1, 0.1]),  # Step 1: Focus on 1.8
        torch.tensor([0.1, 0.0, 0.1, 0.8]),  # Step 2: Focus on 2.1
        torch.tensor([0.8, 0.0, 0.1, 0.0]),  # Step 3: Focus on 5.2
        torch.tensor([0.0, 0.0, 1.0, 0.0]),  # Step 4: Focus on 9.3
    ]
    
    visualize_sorting(input_vals, pointers, attention_weights,
                     save_path="visualizations/demo_sorting.png")
    
    # Demo 2: TSP
    print("\n📦 Demo 2: TSP Tour")
    cities = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.3, 0.7], [0.9, 0.1]])
    tour = torch.tensor([0, 3, 1, 2])  # Visit order
    tour_length = 2.5
    
    visualize_tsp_tour(cities, tour, tour_length=tour_length,
                      save_path="visualizations/demo_tsp.png")
    
    # Demo 3: Convex Hull
    print("\n🎒 Demo 3: Convex Hull")
    points = torch.rand(10, 2)
    hull_indices = torch.tensor([0, 3, 7, 9, 2])  # Boundary points
    
    visualize_convex_hull(points, hull_indices,
                         save_path="visualizations/demo_hull.png")
    
    print("\n✅ All visualizations saved to visualizations/")
    print("=" * 60)
