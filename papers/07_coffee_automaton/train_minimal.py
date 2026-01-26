"""
Day 7: Coffee Automaton - Minimal Training Script

A simple script to run the Coffee Automaton and see complexity rise and fall.
This demonstrates the fundamental pattern: Simple → Complex → Simple

Run this to see complexity dynamics in action!

Author: 30u30 Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from implementation import CoffeeAutomaton, ComplexityTracker, CoffeeExperiments
from visualization import CoffeeVisualizer

def run_basic_experiment():
    """
    Run a basic coffee cooling experiment and visualize results.
    """
    print("🔥 Starting Coffee Automaton: Watching Complexity Bloom and Fade")
    print("=" * 60)
    
    # Run the experiment
    print("☕ Pouring virtual coffee and letting it cool...")
    automaton, tracker = CoffeeExperiments.basic_cooling_experiment(
        size=64, 
        steps=200
    )
    
    # Analyze results
    analysis = tracker.analyze_phases('effective_complexity')
    
    print("\n📊 RESULTS:")
    print(f"   🎯 Peak complexity: {analysis['peak_value']:.4f}")
    print(f"   ⏰ Peak occurred at: step {analysis['peak_time']}")
    print(f"   📈 Growth duration: {analysis['growth_duration']} steps")
    print(f"   📉 Decay duration: {analysis['decay_duration']} steps")
    print(f"   🚀 Growth rate: {analysis['growth_rate']:.6f} units/step")
    print(f"   🌊 Decay rate: {analysis['decay_rate']:.6f} units/step")
    
    # Quick visualization
    print("\n🎨 Creating visualization...")
    visualizer = CoffeeVisualizer()
    
    # Show the main complexity curve
    times, values = tracker.get_complexity_curve('effective_complexity')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, values, linewidth=3, color='#FF6B6B', alpha=0.8)
    ax.fill_between(times, values, alpha=0.3, color='#FF6B6B')
    
    # Mark the peak
    peak_time = analysis['peak_time']
    peak_value = analysis['peak_value']
    ax.scatter([peak_time], [peak_value], color='red', s=150, zorder=5)
    ax.annotate('🎯 PEAK COMPLEXITY\n(Where life thrives!)', 
               xy=(peak_time, peak_value), 
               xytext=(peak_time + 30, peak_value + 0.01),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Add phase labels
    ax.axvspan(0, peak_time, alpha=0.2, color='green', label='🌱 Growth Phase')
    ax.axvspan(peak_time, len(times), alpha=0.2, color='blue', label='🌊 Decay Phase')
    
    ax.set_title('☕ Coffee Automaton: The Universal Pattern of Complexity', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Effective Complexity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add insight text
    insight = """
    💡 KEY INSIGHT:
    Complexity is temporary!
    
    It starts simple, grows complex,
    then returns to simplicity.
    
    This is the universal pattern
    from coffee cooling to the
    evolution of the universe! 🌌
    """
    
    ax.text(0.02, 0.98, insight, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    return automaton, tracker, analysis

def run_life_experiment():
    """
    Run the 'life sweet spot' experiment showing complexity peaks at 
    intermediate energy levels.
    """
    print("\n🌱 Running Life Sweet Spot Experiment...")
    print("Testing: Does complexity peak at intermediate energy levels?")
    
    # Run experiment
    life_results = CoffeeExperiments.life_sweet_spot_experiment()
    
    # Find optimal energy
    optimal_energy = max(life_results.keys(), key=lambda k: life_results[k])
    max_complexity = life_results[optimal_energy]
    
    print(f"\n🎯 LIFE SWEET SPOT FOUND:")
    print(f"   ⚡ Optimal energy level: {optimal_energy:.2f}")
    print(f"   📊 Peak complexity: {max_complexity:.4f}")
    print(f"   💡 This is where life would thrive!")
    
    # Quick plot
    energies = list(life_results.keys())
    complexities = list(life_results.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energies, complexities, linewidth=3, color='#4ECDC4', marker='o', markersize=8)
    ax.fill_between(energies, complexities, alpha=0.3, color='#4ECDC4')
    
    # Mark the sweet spot
    ax.scatter([optimal_energy], [max_complexity], color='red', s=200, zorder=5)
    ax.annotate('🌱 LIFE ZONE\n(Peak Complexity)', 
               xy=(optimal_energy, max_complexity),
               xytext=(optimal_energy + 0.15, max_complexity + 0.01),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Zone markings
    ax.axvspan(0, 0.3, alpha=0.2, color='blue', label='❄️ Too Cold')
    ax.axvspan(0.7, 1.0, alpha=0.2, color='red', label='🔥 Too Hot')
    ax.axvspan(0.3, 0.7, alpha=0.2, color='green', label='🌱 Goldilocks Zone')
    
    ax.set_xlabel('Initial Energy Level', fontsize=12)
    ax.set_ylabel('Peak Complexity Achieved', fontsize=12)
    ax.set_title('🌱 Life Sweet Spot: Complexity vs Energy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return life_results

def main():
    """
    Run the complete Coffee Automaton demonstration.
    """
    print("🌟 WELCOME TO THE COFFEE AUTOMATON 🌟")
    print("Discovering the universal pattern of complexity!")
    print()
    
    # Experiment 1: Basic cooling
    automaton, tracker, analysis = run_basic_experiment()
    
    # Experiment 2: Life sweet spot  
    life_results = run_life_experiment()
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 COFFEE AUTOMATON COMPLETE!")
    print("="*60)
    print()
    print("🔍 WHAT WE DISCOVERED:")
    print(f"   ☕ Coffee cooling follows: Simple → Complex → Simple")
    print(f"   🎯 Peak complexity at step {analysis['peak_time']} with value {analysis['peak_value']:.4f}")
    print(f"   🌱 Life thrives at energy level {max(life_results.keys(), key=lambda k: life_results[k]):.2f}")
    print(f"   🌌 This same pattern governs the entire universe!")
    print()
    print("💡 KEY INSIGHTS:")
    print("   • Complexity is temporary - it rises then falls")
    print("   • Life exists in the complexity peak")  
    print("   • Intelligence emerges in the 'middle' of evolution")
    print("   • This explains why we exist NOW in cosmic history")
    print()
    print("✨ The Coffee Automaton reveals profound truths about:")
    print("   🧠 Intelligence and consciousness")
    print("   🌱 The emergence of life") 
    print("   🌌 The structure of the universe")
    print("   🤖 How AI systems learn and evolve")
    print()
    print("Ready to explore more? Try the full notebook or exercises!")

if __name__ == "__main__":
    main()