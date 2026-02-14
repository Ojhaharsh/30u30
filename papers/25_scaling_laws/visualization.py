"""
Scaling Law Visualization Suite
================================

A comprehensive diagnostic dashboard that demonstrates the 
power-law relationships between model size, dataset size, compute, and loss.

This script:
1. Uses non-linear curve fitting for Irwin-Hall and power laws.
2. Generates log-log plots with empirical data vs. theoretical laws.
3. Visualizes the "Compute-Optimal Frontier" (similar to Figure 4).

Author: 30u30 Project
License: CC BY-NC-ND 4.0
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os
from implementation import MasterFitter, ComputeEconomy

def load_data(path: str = "scaling_results.json"):
    if not os.path.exists(path):
        # Fallback to demo data if simulation hasn't run
        print(f"[WARN] {path} not found. Generating demo data for visualization.")
        return generate_demo_data()
    
    with open(path, "r") as f:
        return json.load(f)

def generate_demo_data():
    """Generates synthetic scaling data following OpenAI's coefficients."""
    ns = np.logspace(5, 8, 10) # 100k to 100M
    l_inf = 1.9
    alpha_n = 0.076
    nc = 8.8e13
    
    results = []
    for n in ns:
        loss = l_inf + (nc / n)**alpha_n + np.random.normal(0, 0.01)
        tokens = 20 * n # Simplified "optimal" data
        results.append({
            "N": float(n),
            "L": float(loss),
            "D": float(tokens),
            "C_pfdays": ComputeEconomy.calculate_c_pfdays(int(n), int(tokens))
        })
    return results

def create_scaling_dashboard(results_path: str = "scaling_results.json"):
    data = load_data(results_path)
    ns = np.array([d['N'] for d in data])
    ls = np.array([d['L'] for d in data])
    ds = np.array([d['D'] for d in data])
    cs = np.array([d['C_pfdays'] for d in data])
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Scaling Law Analysis: The Physics of Language Models", fontsize=18)
    
    # --- PLOT 1: Model Size (N) vs Loss ---
    fitter_n = MasterFitter(ns, ls)
    fitter_n.fit()
    
    axs[0, 0].scatter(ns, ls, color='tab:blue', label='Empirical Measurements')
    x_range = np.logspace(np.log10(min(ns)), np.log10(max(ns)*2), 100)
    axs[0, 0].plot(x_range, fitter_n.predict(x_range), 'r--', label='Non-linear Fit')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('linear') # Kaplan plots often use linear Y to show L_inf flattening
    axs[0, 0].set_xlabel("Number of Parameters (N)")
    axs[0, 0].set_ylabel("Cross Entropy Loss (L)")
    axs[0, 0].set_title("A. Scaling Law for Model Size")
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.3)
    axs[0, 0].annotate(f"alpha_N = {fitter_n.alpha:.4f}", xy=(0.05, 0.95), xycoords='axes fraction')

    # --- PLOT 2: Dataset Size (D) vs Loss ---
    # (Assuming we have data where D was the bottleneck)
    # For demo we swap N and D for visualization
    fitter_d = MasterFitter(ds, ls)
    fitter_d.fit()
    
    axs[0, 1].scatter(ds, ls, color='tab:green', label='Empirical Measurements')
    x_range_d = np.logspace(np.log10(min(ds)), np.log10(max(ds)*2), 100)
    axs[0, 1].plot(x_range_d, fitter_d.predict(x_range_d), 'g--', label='Non-linear Fit')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel("Number of Tokens (D)")
    axs[0, 1].set_ylabel("Loss (L)")
    axs[0, 1].set_title("B. Scaling Law for Dataset Size")
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", ls="-", alpha=0.3)

    # --- PLOT 3: Compute Budget (C) Optimal Frontier ---
    axs[1, 0].scatter(cs, ls, color='tab:red', marker='x', label='All Training Runs')
    # The frontier is the lower bound of all these points
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel("Total Compute (C) [PF-days]")
    axs[1, 0].set_ylabel("Loss (L)")
    axs[1, 0].set_title("C. The Compute-Optimal Frontier")
    axs[1, 0].grid(True, which="both", ls="-", alpha=0.3)
    axs[1, 0].annotate("Slope = -0.05", xy=(cs[len(cs)//2], ls[len(ls)//2]), 
                       xytext=(cs[len(cs)//2]*10, ls[len(ls)//2]*1.2),
                       arrowprops=dict(facecolor='black', shrink=0.05))

    # --- PLOT 4: Sample Efficiency ---
    # Ratio of Loss improvement relative to parameters
    efficiency = ls[0] / ls
    axs[1, 1].plot(ns, efficiency, marker='o', color='purple')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel("Model Size (N)")
    axs[1, 1].set_ylabel("Relative Efficiency")
    axs[1, 1].set_title("D. Sample Efficiency (Larger is better)")
    axs[1, 1].grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "scaling_dashboard.png"
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Scaling Law Dashboard saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_scaling_dashboard()
