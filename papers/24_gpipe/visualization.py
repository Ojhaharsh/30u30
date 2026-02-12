"""
visualization.py - Visualizing GPipe Efficiency

This script generates diagrams to illustrate the core mechanics of GPipe:
1. Pipeline Scheduling (Gantt Chart of Micro-batches)
2. Memory Scaling (Memory vs Micro-batches)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pipeline_schedule(n_partitions, n_microbatches, output_dir='.'):
    """
    Visualizes the synchronous pipeline schedule.
    Ref: Figure 2 from the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # K partitions = Y axis
    # T time steps = X axis
    # Time steps = M + K - 1
    total_steps = n_microbatches + n_partitions - 1
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_microbatches))
    
    for m in range(n_microbatches):
        for k in range(n_partitions):
            # Step when micro-batch m hits partition k
            step = m + k
            ax.barh(n_partitions - 1 - k, 0.8, left=step, color=colors[m], edgecolor='black', alpha=0.8)
            ax.text(step + 0.4, n_partitions - 1 - k, f"M{m}", ha='center', va='center', color='white', fontweight='bold')

    ax.set_yticks(range(n_partitions))
    ax.set_yticklabels([f"Stage {i}" for i in range(n_partitions-1, -1, -1)])
    ax.set_xlabel("Time Step (T)")
    ax.set_title(f"GPipe Synchronous Schedule (K={n_partitions}, M={n_microbatches})")
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'pipeline_schedule.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved pipeline schedule to {path}")

def plot_memory_scaling(n_partitions, hidden_dim, output_dir='.'):
    """
    Illustrates how peak memory scales with micro-batches.
    Equation: Memory ~ (N/M * C) + (K * P)
    Where N/M is micro-batch size and K*P is model param storage.
    """
    microbatch_counts = [1, 2, 4, 8, 16, 32]
    # Synthetic constants for visualization
    activation_size_base = 1000  # MB for batch size 128
    param_size = 200 # MB per stage
    
    mem_no_checkpoint = []
    mem_with_checkpoint = []
    
    for M in microbatch_counts:
        # Without checkpointing: Store all M micro-batch activations at peak?
        # Actually GPipe with sync update stores activations for all M microbatches
        # only if we don't checkpoint. 
        # With checkpointing, we only store at partition boundaries.
        
        # Naive: Mem = (Raw Batch Size) + (Params)
        mem_standard = activation_size_base + (n_partitions * param_size)
        
        # GPipe with Rematerialization:
        # Mem ~ (Batch Size / M) + (Boundary Activations * K) + (Params)
        # Note: GPipe is designed so peak memory is 1/M of naive for activations
        mem_gpipe = (activation_size_base / M) + (n_partitions * param_size) + 50 # Small overhead
        
        mem_no_checkpoint.append(mem_standard)
        mem_with_checkpoint.append(mem_gpipe)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(microbatch_counts, mem_no_checkpoint, 'r--o', label="Standard (No Pipelining)")
    ax.plot(microbatch_counts, mem_with_checkpoint, 'g-s', label="GPipe + Rematerialization")
    
    ax.set_xlabel("Number of Micro-batches (M)")
    ax.set_ylabel("Peak Memory (Arbitrary MB)")
    ax.set_title("Memory Scaling vs Micro-batch Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'memory_scaling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved memory scaling plot to {path}")

if __name__ == "__main__":
    plot_pipeline_schedule(4, 8)
    plot_memory_scaling(4, 512)
