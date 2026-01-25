"""
Minimal Training Script for Complexodynamics

Quick demo of the First Law of Complexodynamics using evolutionary simulation.
Runs fast experiments to demonstrate key concepts:
1. Complexity increases over time
2. Equilibrates at C_max (set by mutation rate)
3. Different organisms reach different equilibria

Usage:
    python train_minimal.py --organism bacteria --generations 10000
    python train_minimal.py --compare --organisms bacteria virus human
    python train_minimal.py --reproduce-figure2

Author: 30u30 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import Dict, List

from implementation import (
    ComplexityTrajectory,
    EvolutionarySimulator,
    shannon_complexity,
    channel_capacity_simple,
    compare_organisms,
    fitness_counting_ones,
    genome_to_string
)


# Organism presets (realistic biological parameters)
ORGANISM_PRESETS = {
    'virus': {
        'name': 'RNA Virus',
        'mutation_rate': 1e-4,
        'genome_length': 10000,
        'selection_strength': 0.05,
        'generation_time': '6 hours',
        'description': 'High mutation, fast replication, low complexity'
    },
    'bacteria': {
        'name': 'E. coli',
        'mutation_rate': 1e-6,
        'genome_length': 4600000,
        'selection_strength': 0.01,
        'generation_time': '20 minutes',
        'description': 'Medium mutation, fast replication, medium complexity'
    },
    'insect': {
        'name': 'Drosophila',
        'mutation_rate': 1e-8,
        'genome_length': 140000000,
        'selection_strength': 0.02,
        'generation_time': '10 days',
        'description': 'Low mutation, slow replication, high complexity'
    },
    'human': {
        'name': 'Homo sapiens',
        'mutation_rate': 1e-9,
        'genome_length': 3200000000,
        'selection_strength': 0.01,
        'generation_time': '25 years',
        'description': 'Very low mutation, very slow replication, very high complexity'
    }
}


def run_basic_experiment(organism: str, generations: int = 10000, verbose: bool = True):
    """
    Run basic complexity evolution experiment for a single organism.
    
    Args:
        organism: One of 'virus', 'bacteria', 'insect', 'human'
        generations: Number of generations to simulate
        verbose: Print detailed progress
        
    Returns:
        Dictionary with results
    """
    if organism not in ORGANISM_PRESETS:
        raise ValueError(f"Unknown organism: {organism}. Choose from {list(ORGANISM_PRESETS.keys())}")
    
    params = ORGANISM_PRESETS[organism]
    
    if verbose:
        print("=" * 70)
        print(f"🧬 COMPLEXODYNAMICS EXPERIMENT: {params['name']}")
        print("=" * 70)
        print(f"Organism: {params['name']}")
        print(f"Description: {params['description']}")
        print(f"Generation time: {params['generation_time']}")
        print(f"Genome length: {params['genome_length']:,} bases")
        print(f"Mutation rate: {params['mutation_rate']:.2e} per base per generation")
        print(f"Selection strength: {params['selection_strength']}")
        print("-" * 70)
    
    # Calculate theoretical maximum complexity
    C_max_theory = -np.log2(params['mutation_rate'] * params['genome_length']) / params['genome_length'] * 1e6
    C_max_theory = min(C_max_theory, 2.0)  # Physical maximum for DNA
    
    if verbose:
        print(f"📊 Theoretical C_max: {C_max_theory:.4f} bits/site")
        print(f"Starting simulation for {generations:,} generations...")
        print()
    
    # Create complexity trajectory simulator
    start_time = time.time()
    
    trajectory = ComplexityTrajectory(
        mutation_rate=params['mutation_rate'],
        genome_length=params['genome_length'],
        selection_strength=params['selection_strength']
    )
    
    # For faster demo, use smaller genome
    demo_genome_length = min(params['genome_length'], 100000)
    
    # Evolve complexity
    sim = EvolutionarySimulator(
        population_size=500,
        genome_length=demo_genome_length,
        mutation_rate=params['mutation_rate'],
        fitness_function=fitness_counting_ones,
        alphabet_size=4  # DNA: A, C, G, T
    )
    
    complexity_history = []
    time_points = []
    
    for gen in range(generations):
        sim.step()
        
        if gen % max(1, generations // 100) == 0:  # Sample 100 points
            C = sim.get_complexity()
            complexity_history.append(C)
            time_points.append(gen)
            
            if verbose and gen % max(1, generations // 10) == 0:
                progress = (gen + 1) / generations * 100
                print(f"  Generation {gen:6,} / {generations:,} ({progress:5.1f}%) | Complexity: {C:.4f} bits/site")
    
    elapsed = time.time() - start_time
    
    # Final results
    C_initial = complexity_history[0] if complexity_history else 0
    C_final = complexity_history[-1] if complexity_history else 0
    C_increase = C_final - C_initial
    
    # Detect equilibrium (when dC/dt ≈ 0)
    equilibrium_gen = None
    threshold = 0.0001
    for i in range(1, len(complexity_history)):
        dC = abs(complexity_history[i] - complexity_history[i-1])
        if dC < threshold:
            equilibrium_gen = time_points[i]
            break
    
    if equilibrium_gen is None:
        equilibrium_gen = generations
    
    if verbose:
        print()
        print("=" * 70)
        print("📈 RESULTS")
        print("=" * 70)
        print(f"Starting complexity:    {C_initial:.4f} bits/site")
        print(f"Final complexity:       {C_final:.4f} bits/site")
        print(f"Complexity increase:    {C_increase:.4f} bits/site ({C_increase/C_initial*100:.1f}%)")
        print(f"Theoretical C_max:      {C_max_theory:.4f} bits/site")
        print(f"Achieved (% of max):    {C_final/C_max_theory*100:.1f}%")
        print(f"Equilibrium reached at: Generation {equilibrium_gen:,}")
        print(f"Simulation time:        {elapsed:.2f} seconds")
        print("=" * 70)
    
    return {
        'organism': params['name'],
        'C_initial': C_initial,
        'C_final': C_final,
        'C_max_theory': C_max_theory,
        'equilibrium_generation': equilibrium_gen,
        'time_points': time_points,
        'complexity_history': complexity_history,
        'elapsed_time': elapsed
    }


def compare_multiple_organisms(organism_list: List[str], generations: int = 5000):
    """
    Compare complexity evolution across multiple organisms.
    
    Args:
        organism_list: List of organism names
        generations: Number of generations to simulate
    """
    print("=" * 70)
    print("🔬 MULTI-ORGANISM COMPARISON")
    print("=" * 70)
    print(f"Comparing {len(organism_list)} organisms over {generations:,} generations")
    print()
    
    results = {}
    
    for organism in organism_list:
        print(f"\n{'='*70}")
        print(f"Running: {organism.upper()}")
        print(f"{'='*70}")
        result = run_basic_experiment(organism, generations, verbose=False)
        results[organism] = result
        
        # Print summary
        print(f"✅ {result['organism']}")
        print(f"   Initial: {result['C_initial']:.4f} → Final: {result['C_final']:.4f} bits/site")
        print(f"   Equilibrium at generation {result['equilibrium_generation']:,}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Organism':<20} {'C_initial':<12} {'C_final':<12} {'C_max':<12} {'Eq. Gen':<12}")
    print("-" * 70)
    
    for organism, result in results.items():
        print(f"{result['organism']:<20} "
              f"{result['C_initial']:<12.4f} "
              f"{result['C_final']:<12.4f} "
              f"{result['C_max_theory']:<12.4f} "
              f"{result['equilibrium_generation']:<12,}")
    
    print("=" * 70)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    for organism, result in results.items():
        params = ORGANISM_PRESETS[organism]
        plt.plot(result['time_points'], result['complexity_history'], 
                 label=result['organism'], linewidth=2)
        plt.axhline(result['C_max_theory'], linestyle='--', alpha=0.3)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Complexity (bits/site)', fontsize=12)
    plt.title('Complexity Evolution: Multi-Organism Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('organism_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: organism_comparison.png")
    plt.show()
    
    return results


def reproduce_figure_2():
    """
    Reproduce Figure 2 from Adami's paper.
    
    Shows complexity vs generations for different mutation rates.
    """
    print("=" * 70)
    print("📄 REPRODUCING PAPER FIGURE 2")
    print("=" * 70)
    print("Plotting complexity trajectories for different mutation rates...")
    print()
    
    mutation_rates = [1e-3, 1e-4, 1e-5, 1e-6]
    genome_length = 1000
    generations = 10000
    
    plt.figure(figsize=(10, 6))
    
    for mu in mutation_rates:
        print(f"  Running μ = {mu:.2e}...")
        
        sim = EvolutionarySimulator(
            population_size=500,
            genome_length=genome_length,
            mutation_rate=mu,
            fitness_function=fitness_counting_ones,
            alphabet_size=2  # Binary for simplicity
        )
        
        complexity_history = []
        
        for gen in range(generations):
            sim.step()
            if gen % 100 == 0:
                complexity_history.append(sim.get_complexity())
        
        time_points = np.arange(0, generations, 100)
        plt.plot(time_points, complexity_history, label=f'μ = {mu:.0e}', linewidth=2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Complexity (bits/site)', fontsize=12)
    plt.title('Figure 2 Reproduction: Complexity vs Mutation Rate', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2_reproduction.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: figure2_reproduction.png")
    plt.show()


def interactive_demo():
    """Interactive command-line demo."""
    print("\n" + "="*70)
    print("🧬 COMPLEXODYNAMICS INTERACTIVE DEMO")
    print("="*70)
    print("\nThis demo shows how complexity evolves according to the")
    print("First Law of Complexodynamics (Adami, 2011)")
    print("\nAvailable organisms:")
    for key, params in ORGANISM_PRESETS.items():
        print(f"  {key:10s} - {params['name']:20s} (μ = {params['mutation_rate']:.0e})")
    
    print("\nPress Enter to run default demo (E. coli, 10000 generations)...")
    print("Or type 'compare' to compare all organisms")
    print("Or type 'figure2' to reproduce paper figure")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'compare':
        compare_multiple_organisms(['virus', 'bacteria', 'insect', 'human'], generations=5000)
    elif choice == 'figure2':
        reproduce_figure_2()
    else:
        run_basic_experiment('bacteria', generations=10000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Minimal demo of the First Law of Complexodynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single organism simulation
  python train_minimal.py --organism bacteria --generations 10000
  
  # Compare multiple organisms
  python train_minimal.py --compare --organisms virus bacteria human
  
  # Reproduce paper figure
  python train_minimal.py --reproduce-figure2
  
  # Interactive mode
  python train_minimal.py --interactive
        """
    )
    
    parser.add_argument('--organism', type=str, choices=['virus', 'bacteria', 'insect', 'human'],
                        help='Organism to simulate')
    parser.add_argument('--generations', type=int, default=10000,
                        help='Number of generations (default: 10000)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple organisms')
    parser.add_argument('--organisms', nargs='+', 
                        choices=['virus', 'bacteria', 'insect', 'human'],
                        help='Organisms to compare (use with --compare)')
    parser.add_argument('--reproduce-figure2', action='store_true',
                        help='Reproduce Figure 2 from the paper')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.interactive:
        interactive_demo()
    elif args.reproduce_figure2:
        reproduce_figure_2()
    elif args.compare:
        organisms = args.organisms if args.organisms else ['virus', 'bacteria', 'insect', 'human']
        compare_multiple_organisms(organisms, args.generations)
    elif args.organism:
        result = run_basic_experiment(args.organism, args.generations)
        
        # Plot trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(result['time_points'], result['complexity_history'], linewidth=2, color='blue')
        plt.axhline(result['C_max_theory'], linestyle='--', color='red', 
                    label=f"C_max (theory) = {result['C_max_theory']:.4f}")
        plt.axvline(result['equilibrium_generation'], linestyle=':', color='green',
                    label=f"Equilibrium at gen {result['equilibrium_generation']:,}")
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Complexity (bits/site)', fontsize=12)
        plt.title(f"Complexity Evolution: {result['organism']}", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.organism}_trajectory.png", dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {args.organism}_trajectory.png")
        plt.show()
    else:
        # No arguments: run default
        print("No arguments provided. Running default demo...")
        print("Use --help to see all options")
        print()
        run_basic_experiment('bacteria', generations=10000)
