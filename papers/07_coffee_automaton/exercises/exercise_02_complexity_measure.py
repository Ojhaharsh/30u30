"""
Exercise 2: Measure Complexity
==============================

Goal: Implement multiple complexity measures and track how they change
as the coffee automaton evolves.

Your Task:
- Implement Shannon entropy
- Implement structure measure (gradient-based)
- Combine into unified complexity metric
- Find the "peak complexity" moment

Learning Objectives:
1. Understand Shannon entropy calculation
2. See how to quantify "structure" in a system
3. Observe the rise-then-fall complexity pattern
4. Understand why complexity peaks in the middle of evolution

Time: 2-3 hours  
Difficulty: Hard ⏱️⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
from exercise_01_build_automaton import CoffeeAutomaton, CoffeeAutomaton1D


def compute_shannon_entropy(state, num_bins=20):
    """
    Compute Shannon entropy of the state.
    
    Shannon entropy measures the "randomness" or "uncertainty" in the data.
    H = -sum(p_i * log(p_i))
    
    Args:
        state: Temperature array (1D or 2D)
        num_bins: Number of bins for histogram
        
    Returns:
        entropy: Shannon entropy value
    """
    # TODO 1: Flatten state if 2D
    flat_state = None  # TODO: state.flatten()
    
    # TODO 2: Create histogram to estimate probability distribution
    # Hint: Use np.histogram with num_bins
    # Handle case where all values are the same
    hist, bin_edges = None, None  # TODO: np.histogram(flat_state, bins=num_bins)
    
    # TODO 3: Convert counts to probabilities
    # Avoid division by zero
    probabilities = None  # TODO: hist / np.sum(hist)
    
    # TODO 4: Compute entropy: H = -sum(p * log(p))
    # Handle p=0 case (0 * log(0) = 0 by convention)
    # Hint: p[p > 0] selects only non-zero probabilities
    entropy = None  # TODO: -np.sum(p * np.log(p + 1e-10)) for p in probabilities
    
    return entropy


def compute_gradient_structure(state):
    """
    Compute a structure measure based on gradients.
    
    High gradients = high structure (sharp boundaries)
    Low gradients = low structure (uniform)
    
    Args:
        state: Temperature array (1D or 2D)
        
    Returns:
        structure: Gradient-based structure measure
    """
    if state.ndim == 1:
        # TODO 5: 1D gradient - difference between adjacent cells
        gradient = None  # TODO: np.abs(np.diff(state))
        structure = None  # TODO: np.mean(gradient)
    else:
        # TODO 6: 2D gradient - use numpy gradient function
        # Compute gradient in x and y directions
        grad_x, grad_y = None, None  # TODO: np.gradient(state)
        gradient_magnitude = None  # TODO: np.sqrt(grad_x**2 + grad_y**2)
        structure = None  # TODO: np.mean(gradient_magnitude)
    
    return structure


def compute_complexity(state, num_bins=20):
    """
    Compute a unified complexity measure.
    
    Key insight: Complexity is highest when there's BOTH
    - High entropy (disorder/randomness)
    - High structure (order/patterns)
    
    Complexity ≈ Entropy × Structure
    
    Or use: Complexity = Entropy × (1 - Entropy) × Structure
    (This peaks when entropy is intermediate)
    
    Args:
        state: Temperature array
        num_bins: Number of bins for entropy calculation
        
    Returns:
        complexity: Unified complexity score
    """
    # TODO 7: Compute entropy
    entropy = None  # TODO: compute_shannon_entropy(state, num_bins)
    
    # TODO 8: Compute structure
    structure = None  # TODO: compute_gradient_structure(state)
    
    # TODO 9: Combine into complexity measure
    # Option 1: Simple product
    # complexity = entropy * structure
    # 
    # Option 2: Normalized (keeps in reasonable range)
    # max_entropy = np.log(num_bins)  # Maximum possible entropy
    # normalized_entropy = entropy / max_entropy
    # complexity = 4 * normalized_entropy * (1 - normalized_entropy) * structure
    
    complexity = None  # TODO: Choose and implement a combination
    
    return complexity


def track_complexity_evolution(automaton, steps=200, measure_interval=1):
    """
    Track complexity over the evolution of an automaton.
    
    Args:
        automaton: CoffeeAutomaton or CoffeeAutomaton1D instance
        steps: Number of simulation steps
        measure_interval: How often to measure (1 = every step)
        
    Returns:
        dict with 'times', 'entropies', 'structures', 'complexities'
    """
    times = []
    entropies = []
    structures = []
    complexities = []
    max_temps = []
    variances = []
    
    for step in range(steps):
        if step % measure_interval == 0:
            state = automaton.grid
            
            # TODO 10: Compute all measures
            entropy = None  # TODO
            structure = None  # TODO
            complexity = None  # TODO
            
            times.append(step)
            entropies.append(entropy)
            structures.append(structure)
            complexities.append(complexity)
            max_temps.append(np.max(state))
            variances.append(np.var(state))
        
        # Step the simulation
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
    """
    Find when complexity peaks.
    
    Args:
        results: Dict from track_complexity_evolution
        
    Returns:
        peak_time: Time step of maximum complexity
        peak_value: Maximum complexity value
    """
    # TODO 11: Find the index of maximum complexity
    peak_idx = None  # TODO: np.argmax(results['complexities'])
    peak_time = None  # TODO: results['times'][peak_idx]
    peak_value = None  # TODO: results['complexities'][peak_idx]
    
    return peak_time, peak_value


def plot_complexity_evolution(results, title="Complexity Evolution"):
    """
    Plot the evolution of entropy, structure, and complexity.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Entropy
    ax1 = axes[0, 0]
    ax1.plot(results['times'], results['entropies'], 'b-', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_title('Entropy Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Structure
    ax2 = axes[0, 1]
    ax2.plot(results['times'], results['structures'], 'g-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Structure (Gradient)')
    ax2.set_title('Structure Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Complexity
    ax3 = axes[1, 0]
    ax3.plot(results['times'], results['complexities'], 'r-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Complexity')
    ax3.set_title('Complexity Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Mark peak complexity
    peak_time, peak_value = find_peak_complexity(results)
    if peak_time is not None:
        ax3.axvline(x=peak_time, color='orange', linestyle='--', 
                   label=f'Peak at t={peak_time}')
        ax3.legend()
    
    # Panel 4: All together (normalized)
    ax4 = axes[1, 1]
    
    # Normalize each to 0-1 range for comparison
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


def test_measures():
    """Test complexity measures."""
    print("Testing Complexity Measures...")
    print("=" * 60)
    
    # Test on known patterns
    print("\n1. Uniform state (low entropy, low structure):")
    uniform = np.ones((50, 50)) * 50
    e = compute_shannon_entropy(uniform)
    s = compute_gradient_structure(uniform)
    c = compute_complexity(uniform)
    print(f"   Entropy: {e:.4f}, Structure: {s:.4f}, Complexity: {c:.4f}")
    
    print("\n2. Random state (high entropy, medium structure):")
    random_state = np.random.uniform(0, 100, (50, 50))
    e = compute_shannon_entropy(random_state)
    s = compute_gradient_structure(random_state)
    c = compute_complexity(random_state)
    print(f"   Entropy: {e:.4f}, Structure: {s:.4f}, Complexity: {c:.4f}")
    
    print("\n3. Single hotspot (low entropy, high structure):")
    hotspot = np.zeros((50, 50))
    hotspot[20:30, 20:30] = 100
    e = compute_shannon_entropy(hotspot)
    s = compute_gradient_structure(hotspot)
    c = compute_complexity(hotspot)
    print(f"   Entropy: {e:.4f}, Structure: {s:.4f}, Complexity: {c:.4f}")
    
    print("\n4. Gradient pattern (medium entropy, high structure):")
    gradient_pattern = np.linspace(0, 100, 50).reshape(1, -1).repeat(50, axis=0)
    e = compute_shannon_entropy(gradient_pattern)
    s = compute_gradient_structure(gradient_pattern)
    c = compute_complexity(gradient_pattern)
    print(f"   Entropy: {e:.4f}, Structure: {s:.4f}, Complexity: {c:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ The pattern with both structure AND variety should")
    print("   have highest complexity!")
    print("=" * 60)


def run_complexity_experiment():
    """Run full complexity tracking experiment."""
    print("\nRunning Complexity Evolution Experiment...")
    print("=" * 60)
    
    # Create automaton with hot spot
    automaton = CoffeeAutomaton(size=50, diffusion_rate=0.15, cooling_rate=0.005)
    automaton.add_hotspot(25, 25, temperature=100, radius=5)
    
    # Track complexity
    results = track_complexity_evolution(automaton, steps=300)
    
    # Find peak
    peak_time, peak_value = find_peak_complexity(results)
    print(f"\nPeak complexity at time step: {peak_time}")
    print(f"Peak complexity value: {peak_value:.4f}")
    
    # Plot results
    plot_complexity_evolution(results, "Coffee Automaton Complexity Evolution")


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. First complete Exercise 1 (build_automaton.py)")
    print("2. Fill in all TODOs in this file")
    print("3. Run test_measures() to verify calculations")
    print("4. Run run_complexity_experiment() for full analysis")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # test_measures()
    # run_complexity_experiment()
