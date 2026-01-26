"""
Solution 2: Measure Complexity
==============================

Complete implementation of Shannon entropy, structure measure, and
unified complexity metric.
"""

import numpy as np
import matplotlib.pyplot as plt
from solution_01_build_automaton import CoffeeAutomaton, CoffeeAutomaton1D


def compute_shannon_entropy(state, num_bins=20):
    """
    Compute Shannon entropy of the state.
    
    H = -sum(p_i * log(p_i))
    """
    flat_state = state.flatten()
    
    # Handle edge case: all same value
    if np.max(flat_state) == np.min(flat_state):
        return 0.0
    
    # Create histogram
    hist, bin_edges = np.histogram(flat_state, bins=num_bins)
    
    # Convert to probabilities
    probabilities = hist / np.sum(hist)
    
    # Compute entropy (handle p=0)
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log(p)
    
    return entropy


def compute_gradient_structure(state):
    """
    Compute structure measure based on gradients.
    High gradients = high structure.
    """
    if state.ndim == 1:
        gradient = np.abs(np.diff(state))
        structure = np.mean(gradient)
    else:
        grad_x, grad_y = np.gradient(state)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        structure = np.mean(gradient_magnitude)
    
    return structure


def compute_complexity(state, num_bins=20):
    """
    Compute unified complexity measure.
    
    Complexity is highest when there's BOTH entropy (variety) AND structure.
    """
    entropy = compute_shannon_entropy(state, num_bins)
    structure = compute_gradient_structure(state)
    
    # Normalized approach
    max_entropy = np.log(num_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Complexity = entropy * (1 - entropy) * structure
    # This peaks when entropy is intermediate
    complexity = 4 * normalized_entropy * (1 - normalized_entropy) * structure
    
    return complexity


def track_complexity_evolution(automaton, steps=200, measure_interval=1):
    """Track complexity over automaton evolution."""
    times = []
    entropies = []
    structures = []
    complexities = []
    max_temps = []
    variances = []
    
    for step in range(steps):
        if step % measure_interval == 0:
            state = automaton.grid
            
            entropy = compute_shannon_entropy(state)
            structure = compute_gradient_structure(state)
            complexity = compute_complexity(state)
            
            times.append(step)
            entropies.append(entropy)
            structures.append(structure)
            complexities.append(complexity)
            max_temps.append(np.max(state))
            variances.append(np.var(state))
        
        automaton.step()
    
    return {
        'times': np.array(times),
        'entropies': np.array(entropies),
        'structures': np.array(structures),
        'complexities': np.array(complexities),
        'max_temps': np.array(max_temps),
        'variances': np.array(variances)
    }


def find_peak_complexity(results):
    """Find when complexity peaks."""
    peak_idx = np.argmax(results['complexities'])
    peak_time = results['times'][peak_idx]
    peak_value = results['complexities'][peak_idx]
    return peak_time, peak_value


def plot_complexity_evolution(results, title="Complexity Evolution"):
    """Plot evolution of all measures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Entropy
    ax1 = axes[0, 0]
    ax1.plot(results['times'], results['entropies'], 'b-', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_title('Entropy Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Structure
    ax2 = axes[0, 1]
    ax2.plot(results['times'], results['structures'], 'g-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Structure (Gradient)')
    ax2.set_title('Structure Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Complexity
    ax3 = axes[1, 0]
    ax3.plot(results['times'], results['complexities'], 'r-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Complexity')
    ax3.set_title('Complexity Over Time')
    ax3.grid(True, alpha=0.3)
    
    peak_time, peak_value = find_peak_complexity(results)
    ax3.axvline(x=peak_time, color='orange', linestyle='--', 
               label=f'Peak at t={peak_time}')
    ax3.legend()
    
    # All normalized
    ax4 = axes[1, 1]
    
    def normalize(arr):
        if np.max(arr) - np.min(arr) > 0:
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return arr
    
    ax4.plot(results['times'], normalize(results['entropies']), 
            'b-', linewidth=2, label='Entropy')
    ax4.plot(results['times'], normalize(results['structures']), 
            'g-', linewidth=2, label='Structure')
    ax4.plot(results['times'], normalize(results['complexities']), 
            'r-', linewidth=2, label='Complexity')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('All Measures (Normalized)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def demo():
    """Demonstrate complexity measures."""
    print("Complexity Measures Demo")
    print("=" * 60)
    
    # Test on patterns
    print("\n1. Testing on known patterns:")
    
    patterns = {
        'Uniform': np.ones((50, 50)) * 50,
        'Random': np.random.uniform(0, 100, (50, 50)),
        'Single hotspot': np.zeros((50, 50)),
        'Gradient': np.linspace(0, 100, 50).reshape(1, -1).repeat(50, axis=0)
    }
    patterns['Single hotspot'][20:30, 20:30] = 100
    
    for name, pattern in patterns.items():
        e = compute_shannon_entropy(pattern)
        s = compute_gradient_structure(pattern)
        c = compute_complexity(pattern)
        print(f"   {name:20s}: E={e:.3f}, S={s:.3f}, C={c:.3f}")
    
    # Evolution experiment
    print("\n2. Running evolution experiment...")
    automaton = CoffeeAutomaton(size=50, diffusion_rate=0.15, cooling_rate=0.005)
    automaton.add_hotspot(25, 25, temperature=100, radius=5)
    
    results = track_complexity_evolution(automaton, steps=300)
    
    peak_time, peak_value = find_peak_complexity(results)
    print(f"\n   Peak complexity at time: {peak_time}")
    print(f"   Peak value: {peak_value:.4f}")
    
    plot_complexity_evolution(results)
    

if __name__ == "__main__":
    demo()
