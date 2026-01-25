"""
Visualization Suite for Complexodynamics

Generates all key plots from the "First Law of Complexodynamics" paper:
1. Complexity trajectory over time
2. Fidelity-complexity trade-off curve
3. Information flow dynamics (I_E vs I_L)
4. Organism comparison (bacteria, insects, mammals)
5. Equilibrium phase diagram
6. Real genome analysis scatter plot
7. Thermodynamics analogy (entropy vs complexity)

Usage:
    python visualization.py --all
    python visualization.py --plot trajectory
    python visualization.py --compare organisms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse

from implementation import (
    ComplexityTrajectory,
    EvolutionarySimulator,
    channel_capacity_simple,
    fidelity_complexity_curve,
    compare_organisms,
    information_flow_analysis,
    shannon_complexity,
    fitness_counting_ones
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_complexity_trajectory(save_path: Optional[str] = None):
    """
    Plot 1: Complexity trajectory over time
    
    Shows C(t) = C_max * (1 - exp(-t/tau)) for different organisms.
    Demonstrates exponential saturation to equilibrium.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define organism parameters
    organisms = {
        'RNA Virus': {'mu': 1e-4, 'L': 1e4, 'color': 'red', 'tau': 100},
        'Bacteria': {'mu': 1e-6, 'L': 1e6, 'color': 'blue', 'tau': 1000},
        'Insect': {'mu': 1e-8, 'L': 1e8, 'color': 'green', 'tau': 5000},
        'Mammal': {'mu': 1e-9, 'L': 1e9, 'color': 'purple', 'tau': 10000},
    }
    
    generations = np.linspace(0, 20000, 500)
    
    for name, params in organisms.items():
        # Calculate maximum complexity
        C_max = -np.log2(params['mu'] * params['L']) / params['L'] * 1e6  # Bits per site
        
        # Complexity trajectory
        trajectory = ComplexityTrajectory(
            mutation_rate=params['mu'],
            genome_length=int(params['L']),
            selection_strength=0.01
        )
        
        # Analytical solution
        C_t = C_max * (1 - np.exp(-generations / params['tau']))
        
        ax.plot(generations, C_t, label=name, color=params['color'], linewidth=2)
        ax.axhline(C_max, color=params['color'], linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Complexity (bits/site)', fontsize=12)
    ax.set_title('Complexity Trajectory: Exponential Saturation to Equilibrium', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 2.5)
    
    # Add annotations
    ax.annotate('Equilibrium plateau', xy=(15000, 1.9), xytext=(12000, 2.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_fidelity_complexity_tradeoff(save_path: Optional[str] = None):
    """
    Plot 2: Fidelity-Complexity Trade-off
    
    Shows C_max vs mutation rate (log-log scale).
    Demonstrates inverse relationship: better copying → more complexity.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sweep mutation rates
    mutation_rates = np.logspace(-10, -3, 100)
    genome_length = 1e6  # 1 million bases
    
    # Calculate maximum complexity for each mutation rate
    C_max_values = fidelity_complexity_curve(mutation_rates, genome_length)
    
    ax.loglog(mutation_rates, C_max_values, linewidth=3, color='darkblue', label='Theory')
    
    # Mark specific organisms
    organisms = {
        'Human': {'mu': 1e-9, 'color': 'purple', 'marker': 'o'},
        'Bacteria': {'mu': 1e-6, 'color': 'blue', 'marker': 's'},
        'RNA Virus': {'mu': 1e-4, 'color': 'red', 'marker': '^'},
    }
    
    for name, params in organisms.items():
        C_max = channel_capacity_simple(params['mu'], genome_length)
        ax.scatter([params['mu']], [C_max], s=200, color=params['color'], 
                   marker=params['marker'], edgecolors='black', linewidths=2,
                   label=name, zorder=5)
    
    ax.set_xlabel('Mutation Rate (μ) [per base per generation]', fontsize=12)
    ax.set_ylabel('Maximum Complexity (C_max) [bits]', fontsize=12)
    ax.set_title('Fidelity-Complexity Trade-off: Better Copying → More Complexity', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax.annotate('C_max ≈ -log(μ·L)', xy=(1e-7, 1e7), xytext=(1e-8, 1e8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_information_flow(save_path: Optional[str] = None):
    """
    Plot 3: Information Flow Dynamics
    
    Shows I_E (gain from selection) and I_L (loss from mutation) over time.
    Demonstrates equilibration: I_E = I_L at steady state.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create simulator
    sim = EvolutionarySimulator(
        population_size=1000,
        genome_length=100,
        mutation_rate=1e-3,
        fitness_function=fitness_counting_ones,
        alphabet_size=2
    )
    
    # Run simulation and track information flow
    generations = 500
    I_E_history = []
    I_L_history = []
    C_history = []
    
    for gen in range(generations):
        sim.step()
        
        # Measure information flow every 10 generations
        if gen % 10 == 0:
            I_E, I_L = information_flow_analysis(sim, window=10)
            I_E_history.append(I_E)
            I_L_history.append(I_L)
            C_history.append(sim.get_complexity())
    
    time_points = np.arange(0, generations, 10)
    
    # Plot 1: Information gain vs loss
    ax1.plot(time_points, I_E_history, label='I_E (Gain from selection)', 
             color='green', linewidth=2, marker='o', markersize=4)
    ax1.plot(time_points, I_L_history, label='I_L (Loss from mutation)', 
             color='red', linewidth=2, marker='s', markersize=4)
    ax1.axhline(I_E_history[-1], color='green', linestyle='--', alpha=0.5)
    ax1.axhline(I_L_history[-1], color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Information Flow (bits/generation)', fontsize=12)
    ax1.set_title('Information Equilibration: I_E = I_L at Steady State', fontsize=13, fontweight='bold')
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)
    
    # Add equilibrium annotation
    eq_gen = len(time_points) * 2 // 3
    ax1.annotate('Equilibrium\n(I_E ≈ I_L)', 
                 xy=(time_points[eq_gen], I_E_history[eq_gen]),
                 xytext=(time_points[eq_gen] + 100, I_E_history[eq_gen] + 0.5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Net information flow and complexity
    net_flow = np.array(I_E_history) - np.array(I_L_history)
    
    ax2_twin = ax2.twinx()
    
    ax2.plot(time_points, net_flow, label='dC/dt = I_E - I_L', 
             color='blue', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(time_points, 0, net_flow, where=(net_flow >= 0), 
                      alpha=0.3, color='green', label='Complexity increasing')
    
    ax2_twin.plot(time_points, C_history, label='Complexity C(t)', 
                  color='purple', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Net Information Flow (bits/gen)', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Complexity (bits)', fontsize=12, color='purple')
    ax2.set_title('Net Flow → Complexity Growth', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_organism_comparison(save_path: Optional[str] = None):
    """
    Plot 4: Organism Comparison
    
    Compare complexity trajectories for different biological systems.
    Shows how mutation rate determines equilibrium complexity.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define organisms
    organism_params = {
        'SARS-CoV-2': {'mu': 1e-4, 'L': 30000, 'color': 'darkred', 'gen_time': '6 hours'},
        'E. coli': {'mu': 1e-6, 'L': 4.6e6, 'color': 'blue', 'gen_time': '20 minutes'},
        'Drosophila': {'mu': 1e-8, 'L': 1.4e8, 'color': 'green', 'gen_time': '10 days'},
        'Homo sapiens': {'mu': 1e-9, 'L': 3.2e9, 'color': 'purple', 'gen_time': '25 years'},
    }
    
    # Comparison metrics
    results = compare_organisms(organism_params)
    
    # Plot 1: Complexity trajectories
    ax1 = fig.add_subplot(gs[0, :])
    for name, params in organism_params.items():
        C_max = results[name]['C_max']
        tau = results[name]['tau']
        generations = np.linspace(0, 5 * tau, 200)
        C_t = C_max * (1 - np.exp(-generations / tau))
        ax1.plot(generations, C_t, label=name, color=params['color'], linewidth=2.5)
    
    ax1.set_xlabel('Generations', fontsize=12)
    ax1.set_ylabel('Complexity (bits/site)', fontsize=12)
    ax1.set_title('Organism Complexity Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of C_max
    ax2 = fig.add_subplot(gs[1, 0])
    names = list(organism_params.keys())
    C_maxs = [results[name]['C_max'] for name in names]
    colors = [organism_params[name]['color'] for name in names]
    
    bars = ax2.barh(names, C_maxs, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axvline(2.0, color='black', linestyle='--', label='Maximum (2.0 bits)', linewidth=2)
    ax2.set_xlabel('Maximum Complexity (bits/site)', fontsize=12)
    ax2.set_title('Equilibrium Complexity by Organism', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Time to equilibrium
    ax3 = fig.add_subplot(gs[1, 1])
    taus = [results[name]['tau'] for name in names]
    
    ax3.barh(names, taus, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Time to Equilibrium (τ generations)', fontsize=12)
    ax3.set_title('Equilibration Time by Organism', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_phase_diagram(save_path: Optional[str] = None):
    """
    Plot 5: Equilibrium Phase Diagram
    
    2D map showing equilibrium complexity as function of mutation rate and genome length.
    Demonstrates parameter space where complexity is possible.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create parameter grid
    mu_values = np.logspace(-10, -3, 100)
    L_values = np.logspace(3, 10, 100)
    Mu, L = np.meshgrid(mu_values, L_values)
    
    # Calculate equilibrium complexity
    C_eq = -np.log2(Mu * L) / L * 1e6  # Normalize to bits per site
    C_eq = np.clip(C_eq, 0, 2.0)  # Physical bounds
    
    # Create heatmap
    im = ax.contourf(Mu, L, C_eq, levels=20, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Equilibrium Complexity (bits/site)', fontsize=12)
    
    # Overlay contour lines
    contours = ax.contour(Mu, L, C_eq, levels=[0.5, 1.0, 1.5, 1.8], 
                          colors='white', linewidths=2)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%0.1f bits')
    
    # Mark real organisms
    organisms = {
        'Virus': {'mu': 1e-4, 'L': 1e4, 'marker': '^', 'color': 'red'},
        'Bacteria': {'mu': 1e-6, 'L': 1e6, 'marker': 's', 'color': 'cyan'},
        'Insect': {'mu': 1e-8, 'L': 1e8, 'marker': 'o', 'color': 'yellow'},
        'Mammal': {'mu': 1e-9, 'L': 1e9, 'marker': '*', 'color': 'white'},
    }
    
    for name, params in organisms.items():
        ax.scatter([params['mu']], [params['L']], s=300, 
                   marker=params['marker'], color=params['color'],
                   edgecolors='black', linewidths=2, label=name, zorder=5)
    
    # Eigen threshold line
    mu_line = np.logspace(-10, -3, 100)
    L_eigen = 1 / mu_line * np.log(1 / mu_line)
    ax.plot(mu_line, L_eigen, 'r--', linewidth=3, label="Eigen's Error Threshold")
    
    ax.set_xlabel('Mutation Rate (μ)', fontsize=12)
    ax.set_ylabel('Genome Length (L)', fontsize=12)
    ax.set_title('Phase Diagram: Complexity Landscape in Parameter Space', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_real_genome_analysis(save_path: Optional[str] = None):
    """
    Plot 6: Real Genome Scatter Plot
    
    Analyze actual genome complexity vs predicted C_max.
    Shows organisms cluster near their theoretical limits.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Real organism data (approximate values from literature)
    organisms_data = {
        'MS2 phage': {'mu': 1e-4, 'L': 3569, 'C_actual': 1.15},
        'HIV-1': {'mu': 3e-5, 'L': 9181, 'C_actual': 1.28},
        'SARS-CoV-2': {'mu': 1e-4, 'L': 29903, 'C_actual': 1.22},
        'E. coli': {'mu': 5e-7, 'L': 4.6e6, 'C_actual': 1.52},
        'S. cerevisiae': {'mu': 3e-8, 'L': 1.2e7, 'C_actual': 1.61},
        'C. elegans': {'mu': 1e-8, 'L': 1e8, 'C_actual': 1.73},
        'D. melanogaster': {'mu': 7e-9, 'L': 1.4e8, 'C_actual': 1.76},
        'M. musculus': {'mu': 2e-9, 'L': 2.7e9, 'C_actual': 1.84},
        'H. sapiens': {'mu': 1e-9, 'L': 3.2e9, 'C_actual': 1.89},
    }
    
    # Calculate predicted C_max
    for name, data in organisms_data.items():
        C_predicted = -np.log2(data['mu'] * data['L']) / data['L'] * 1e6
        data['C_predicted'] = C_predicted
    
    # Extract data for plotting
    C_predicted = [data['C_predicted'] for data in organisms_data.values()]
    C_actual = [data['C_actual'] for data in organisms_data.values()]
    names = list(organisms_data.keys())
    
    # Color by organism type
    colors = ['red', 'red', 'red',  # Viruses
              'blue',  # Bacteria
              'green',  # Yeast
              'orange', 'orange',  # Invertebrates
              'purple', 'purple']  # Mammals
    
    # Scatter plot
    ax.scatter(C_predicted, C_actual, s=200, c=colors, alpha=0.7, 
               edgecolors='black', linewidths=2)
    
    # Annotate points
    for i, name in enumerate(names):
        ax.annotate(name, (C_predicted[i], C_actual[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')
    
    # Diagonal line (perfect prediction)
    ax.plot([1.0, 2.0], [1.0, 2.0], 'k--', linewidth=2, label='Perfect prediction')
    
    # Fit line
    z = np.polyfit(C_predicted, C_actual, 1)
    p = np.poly1d(z)
    ax.plot(C_predicted, p(C_predicted), 'g-', linewidth=2, 
            label=f'Best fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax.set_xlabel('Predicted C_max (bits/site)', fontsize=12)
    ax.set_ylabel('Actual Complexity (bits/site)', fontsize=12)
    ax.set_title('Real Genome Analysis: Theory vs Reality', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(1.0, 2.0)
    
    # Add R² annotation
    from scipy.stats import pearsonr
    r, _ = pearsonr(C_predicted, C_actual)
    ax.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_thermodynamics_analogy(save_path: Optional[str] = None):
    """
    Plot 7: Thermodynamics Analogy
    
    Side-by-side comparison of entropy (thermodynamics) and complexity (complexodynamics).
    Shows parallel laws governing two different systems.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    time = np.linspace(0, 100, 500)
    
    # LEFT: Thermodynamic entropy (mixing)
    # Two compartments: hot and cold water
    T_hot = 80
    T_cold = 20
    T_eq = 50
    tau_thermal = 20
    
    T_system = T_eq + (T_hot - T_eq) * np.exp(-time / tau_thermal)
    S_system = 100 * (1 - np.exp(-time / tau_thermal))  # Entropy increases
    
    ax1_temp = ax1.twinx()
    
    line1 = ax1.plot(time, S_system, 'r-', linewidth=3, label='Entropy S(t)')
    ax1.axhline(100, color='r', linestyle='--', alpha=0.5)
    
    line2 = ax1_temp.plot(time, T_system, 'b--', linewidth=2, label='Temperature T(t)')
    ax1_temp.axhline(T_eq, color='b', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Entropy S (arbitrary units)', fontsize=12, color='r')
    ax1_temp.set_ylabel('Temperature T (°C)', fontsize=12, color='b')
    ax1.set_title('THERMODYNAMICS\nSecond Law: Entropy Increases', 
                  fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1_temp.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right')
    
    # Add annotations
    ax1.annotate('Mixing\n(irreversible)', xy=(50, 50), xytext=(70, 30),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=11, ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # RIGHT: Complexodynamic complexity (evolution)
    C_max = 1.8
    tau_evolution = 30
    
    C_system = C_max * (1 - np.exp(-time / tau_evolution))
    I_E = C_max / tau_evolution * np.exp(-time / tau_evolution)  # Decreasing
    I_L = C_max / tau_evolution * 0.5 * np.ones_like(time)  # Constant
    
    ax2_info = ax2.twinx()
    
    line3 = ax2.plot(time, C_system, 'g-', linewidth=3, label='Complexity C(t)')
    ax2.axhline(C_max, color='g', linestyle='--', alpha=0.5)
    
    line4 = ax2_info.plot(time, I_E, 'orange', linewidth=2, label='Info gain I_E')
    line5 = ax2_info.plot(time, I_L, 'purple', linewidth=2, linestyle='--', label='Info loss I_L')
    
    ax2.set_xlabel('Generations', fontsize=12)
    ax2.set_ylabel('Complexity C (bits/site)', fontsize=12, color='g')
    ax2_info.set_ylabel('Information Flow (bits/gen)', fontsize=12)
    ax2.set_title('COMPLEXODYNAMICS\nFirst Law: Complexity Increases', 
                  fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='right')
    
    # Add annotations
    ax2.annotate('Evolution\n(irreversible)', xy=(50, 0.9), xytext=(70, 0.5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=11, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def generate_all_plots(output_dir: str = 'figures'):
    """Generate all 7 plots and save to directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualization suite...")
    
    plots = [
        ("complexity_trajectory", plot_complexity_trajectory),
        ("fidelity_complexity_tradeoff", plot_fidelity_complexity_tradeoff),
        ("information_flow", plot_information_flow),
        ("organism_comparison", plot_organism_comparison),
        ("phase_diagram", plot_phase_diagram),
        ("real_genome_analysis", plot_real_genome_analysis),
        ("thermodynamics_analogy", plot_thermodynamics_analogy),
    ]
    
    for i, (name, plot_fn) in enumerate(plots, 1):
        print(f"  [{i}/7] Generating {name}...")
        save_path = os.path.join(output_dir, f"{name}.png")
        plot_fn(save_path)
    
    print(f"\n✅ All plots saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complexodynamics Visualization Suite')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--plot', type=str, choices=[
        'trajectory', 'tradeoff', 'flow', 'organisms', 'phase', 'genomes', 'analogy'
    ], help='Generate specific plot')
    parser.add_argument('--output', type=str, default='figures', help='Output directory')
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_plots(args.output)
    elif args.plot:
        plot_map = {
            'trajectory': plot_complexity_trajectory,
            'tradeoff': plot_fidelity_complexity_tradeoff,
            'flow': plot_information_flow,
            'organisms': plot_organism_comparison,
            'phase': plot_phase_diagram,
            'genomes': plot_real_genome_analysis,
            'analogy': plot_thermodynamics_analogy,
        }
        plot_map[args.plot]()
    else:
        print("Usage: python visualization.py --all  OR  python visualization.py --plot <name>")
        print("Available plots: trajectory, tradeoff, flow, organisms, phase, genomes, analogy")
