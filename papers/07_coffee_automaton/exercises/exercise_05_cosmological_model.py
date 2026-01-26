"""
Exercise 5: Cosmological Model
==============================

Goal: Build a toy model of cosmic evolution showing complexity peaking 
during the "structure formation era."

Your Task:
- Create a particle simulation with expansion and gravity
- Measure complexity as particles cluster
- Show the three eras: hot uniform soup → structure formation → maximum entropy
- Understand why life exists "now" in cosmic history

Learning Objectives:
1. Toy cosmology simulation
2. How structure forms from uniform beginnings
3. Why complexity must peak and then decline
4. The "Goldilocks zone" for life

Time: 4-5 hours
Difficulty: Expert ⏱️⏱️⏱️⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import complexity measures
try:
    from exercise_02_complexity_measure import compute_shannon_entropy
except ImportError:
    def compute_shannon_entropy(x, num_bins=20):
        flat = x.flatten()
        hist, _ = np.histogram(flat, bins=num_bins)
        p = hist / np.sum(hist)
        return -np.sum(p[p > 0] * np.log(p[p > 0]))


class ToyUniverse:
    """
    A toy model of cosmic evolution.
    
    Physics (simplified):
    - Particles attract via gravity (1/r² force)
    - Universe expands over time (scale factor increases)
    - Particles have thermal motion (velocity noise)
    - Hubble friction slows particles as universe expands
    """
    
    def __init__(self, n_particles=200, box_size=100.0, 
                 gravity_strength=0.1, thermal_strength=0.5,
                 expansion_rate=0.001):
        """
        Initialize the universe.
        
        Args:
            n_particles: Number of matter particles
            box_size: Initial size of the universe
            gravity_strength: How strong gravity is
            thermal_strength: How much random motion (temperature)
            expansion_rate: How fast the universe expands (Hubble constant analog)
        """
        self.n_particles = n_particles
        self.box_size = box_size
        self.gravity_strength = gravity_strength
        self.thermal_strength = thermal_strength
        self.expansion_rate = expansion_rate
        
        # TODO 1: Initialize particle positions uniformly
        # Particles start distributed throughout the box
        self.positions = None  # TODO: np.random.uniform(0, box_size, (n_particles, 2))
        
        # TODO 2: Initialize velocities randomly (thermal motion)
        self.velocities = None  # TODO: np.random.randn(n_particles, 2) * thermal_strength
        
        # Track time and scale factor
        self.time = 0.0
        self.scale_factor = 1.0  # Universe size relative to initial
        
        # History for analysis
        self.history = {
            'time': [],
            'scale_factor': [],
            'positions': [],
            'complexity': [],
            'clustering': [],
            'entropy': []
        }
        
    def compute_gravity(self):
        """
        Compute gravitational accelerations for all particles.
        
        Gravity: F ∝ 1/r² toward each other particle
        """
        accelerations = np.zeros_like(self.positions)
        
        # TODO 3: Compute pairwise gravitational forces
        # For each pair of particles, compute attraction
        # Use softening to prevent division by zero
        
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i == j:
                    continue
                
                # Vector from i to j
                r = None  # TODO: self.positions[j] - self.positions[i]
                
                # Distance with softening
                dist = None  # TODO: np.sqrt(np.sum(r**2) + 1.0)  # +1 for softening
                
                # Gravitational acceleration
                # a = G * M / r^2 * direction
                acc = None  # TODO: self.gravity_strength * r / (dist**3)
                
                accelerations[i] += acc if acc is not None else 0
        
        return accelerations
    
    def compute_thermal_kick(self):
        """Random thermal motion (temperature)."""
        # TODO 4: Add random velocity kicks
        # This represents thermal energy / radiation pressure
        # Decreases as universe expands (temperature drops)
        
        temperature = None  # TODO: self.thermal_strength / (self.scale_factor ** 0.5)
        kicks = None  # TODO: np.random.randn(self.n_particles, 2) * temperature
        
        return kicks if kicks is not None else np.zeros_like(self.velocities)
    
    def step(self, dt=1.0):
        """
        Evolve the universe by one time step.
        
        Physics simulation:
        1. Expand the universe (increase scale factor)
        2. Compute gravity
        3. Add thermal motion
        4. Update velocities and positions
        5. Apply boundary conditions
        """
        self.time += dt
        
        # TODO 5: Universe expansion
        # Scale factor increases over time (like real universe)
        old_scale = self.scale_factor
        self.scale_factor = None  # TODO: 1.0 + self.expansion_rate * self.time
        
        # Hubble drag: velocities decrease as universe expands
        # v_physical = v_comoving / scale_factor
        scale_ratio = old_scale / self.scale_factor if self.scale_factor else 1.0
        self.velocities *= scale_ratio
        
        # TODO 6: Compute and apply gravitational acceleration
        gravity_acc = self.compute_gravity()
        self.velocities += gravity_acc * dt
        
        # TODO 7: Add thermal kicks
        thermal_kicks = self.compute_thermal_kick()
        self.velocities += thermal_kicks * np.sqrt(dt)
        
        # TODO 8: Update positions
        self.positions += self.velocities * dt
        
        # Periodic boundary conditions (wrap around)
        current_box = self.box_size * self.scale_factor
        self.positions = self.positions % current_box
        
        # Record state
        self._record_state()
    
    def _record_state(self):
        """Record current state for analysis."""
        self.history['time'].append(self.time)
        self.history['scale_factor'].append(self.scale_factor)
        self.history['positions'].append(self.positions.copy())
        
        # Compute complexity measures
        complexity = self.compute_complexity()
        self.history['complexity'].append(complexity['total'])
        self.history['clustering'].append(complexity['clustering'])
        self.history['entropy'].append(complexity['entropy'])
    
    def compute_complexity(self):
        """
        Compute complexity of current particle distribution.
        
        Key insight: 
        - Early universe: uniform (low complexity)
        - Middle: clusters forming (high complexity)
        - Late: everything in few clusters (low complexity again)
        """
        # TODO 9: Compute clustering measure
        # How "clumpy" is the distribution?
        # One approach: variance of local density
        
        # Divide box into grid cells and count particles per cell
        n_cells = 10
        current_box = self.box_size * self.scale_factor
        cell_size = current_box / n_cells
        
        counts = np.zeros((n_cells, n_cells))
        for pos in self.positions:
            i = int(pos[0] / cell_size) % n_cells
            j = int(pos[1] / cell_size) % n_cells
            counts[i, j] += 1
        
        # Clustering = variance of counts (normalized)
        clustering = np.var(counts) / (np.mean(counts) + 1e-10)
        
        # TODO 10: Compute spatial entropy
        # How evenly distributed are particles?
        entropy = compute_shannon_entropy(counts)
        
        # TODO 11: Combine into complexity
        # Complexity is high when there's structure (clustering)
        # but also variety (entropy not too low)
        
        # Normalized entropy (0-1 scale)
        max_entropy = np.log(n_cells * n_cells)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Complexity peaks at intermediate values
        # Use: complexity = clustering * entropy * (1 - entropy)
        # This is maximized when entropy is around 0.5 and clustering is high
        
        complexity = None  # TODO: clustering * norm_entropy * (1 - norm_entropy) * 4
        
        return {
            'clustering': clustering,
            'entropy': norm_entropy,
            'total': complexity if complexity else 0
        }
    
    def run(self, steps=500, dt=1.0):
        """Run simulation for given number of steps."""
        for _ in range(steps):
            self.step(dt)


def visualize_universe(universe, interval=50):
    """Create animation of cosmic evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    initial_box = universe.box_size
    sc = ax1.scatter(universe.positions[:, 0], universe.positions[:, 1], 
                    s=10, c='white', alpha=0.8)
    ax1.set_facecolor('black')
    ax1.set_xlim(0, initial_box * 3)
    ax1.set_ylim(0, initial_box * 3)
    ax1.set_title('Universe Evolution')
    ax1.set_aspect('equal')
    
    line, = ax2.plot([], [], 'r-', linewidth=2)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 5)
    ax2.set_xlabel('Cosmic Time')
    ax2.set_ylabel('Complexity')
    ax2.set_title('Complexity Over Cosmic Time')
    ax2.grid(True, alpha=0.3)
    
    def update(frame):
        universe.step()
        
        sc.set_offsets(universe.positions)
        current_box = universe.box_size * universe.scale_factor
        ax1.set_xlim(0, current_box)
        ax1.set_ylim(0, current_box)
        ax1.set_title(f'Time: {universe.time:.0f} | Scale: {universe.scale_factor:.2f}')
        
        times = universe.history['time']
        complexities = universe.history['complexity']
        line.set_data(times, complexities)
        ax2.set_xlim(0, max(times) + 10)
        ax2.set_ylim(0, max(complexities) * 1.2 + 0.1)
        
        return [sc, line]
    
    ani = FuncAnimation(fig, update, frames=500, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


def analyze_cosmic_evolution():
    """
    Analyze the full cosmic evolution and complexity curve.
    """
    print("Running Cosmic Evolution Simulation...")
    print("=" * 60)
    
    # Create universe
    universe = ToyUniverse(
        n_particles=300,
        box_size=100.0,
        gravity_strength=0.05,
        thermal_strength=0.3,
        expansion_rate=0.002
    )
    
    # Run simulation
    universe.run(steps=1000, dt=1.0)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = np.array(universe.history['time'])
    
    # Panel 1: Scale factor
    ax1 = axes[0, 0]
    ax1.plot(times, universe.history['scale_factor'], 'b-', linewidth=2)
    ax1.set_xlabel('Cosmic Time')
    ax1.set_ylabel('Scale Factor')
    ax1.set_title('Universe Expansion')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Clustering
    ax2 = axes[0, 1]
    ax2.plot(times, universe.history['clustering'], 'g-', linewidth=2)
    ax2.set_xlabel('Cosmic Time')
    ax2.set_ylabel('Clustering')
    ax2.set_title('Structure Formation (Clustering)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Entropy
    ax3 = axes[1, 0]
    ax3.plot(times, universe.history['entropy'], 'm-', linewidth=2)
    ax3.set_xlabel('Cosmic Time')
    ax3.set_ylabel('Spatial Entropy')
    ax3.set_title('Entropy Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Complexity
    ax4 = axes[1, 1]
    complexities = np.array(universe.history['complexity'])
    ax4.plot(times, complexities, 'r-', linewidth=2)
    ax4.set_xlabel('Cosmic Time')
    ax4.set_ylabel('Complexity')
    ax4.set_title('Cosmic Complexity')
    ax4.grid(True, alpha=0.3)
    
    # Mark peak complexity
    peak_idx = np.argmax(complexities)
    peak_time = times[peak_idx]
    ax4.axvline(x=peak_time, color='orange', linestyle='--', 
               label=f'Peak at t={peak_time:.0f}')
    ax4.legend()
    
    # Add era labels
    ax4.annotate('Early Universe\n(Hot, Uniform)', 
                xy=(times[0], complexities[0]), fontsize=10,
                xytext=(times[0] + 100, complexities[0] + 1),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax4.annotate('Structure Formation\n(Life possible!)', 
                xy=(peak_time, complexities[peak_idx]), fontsize=10,
                xytext=(peak_time + 100, complexities[peak_idx] + 0.5),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax4.annotate('Heat Death\n(All in clusters)', 
                xy=(times[-1], complexities[-1]), fontsize=10,
                xytext=(times[-1] - 200, complexities[-1] + 1),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.suptitle('Toy Cosmological Model: Complexity Evolution', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Show snapshots
    show_snapshots(universe)
    
    print(f"\n🎯 Peak complexity at cosmic time: {peak_time:.0f}")
    print(f"   This is when 'life' would be possible!")
    print("\nThe three eras:")
    print("1. EARLY: Hot, uniform soup - too chaotic for structure")
    print("2. MIDDLE: Structure formation - complexity peaks (we are here)")
    print("3. LATE: Maximum entropy - everything in dead clusters")


def show_snapshots(universe):
    """Show snapshots at different cosmic times."""
    times = np.array(universe.history['time'])
    positions = universe.history['positions']
    complexities = np.array(universe.history['complexity'])
    
    # Select representative snapshots
    early_idx = len(times) // 10
    peak_idx = np.argmax(complexities)
    late_idx = len(times) - 1
    
    indices = [early_idx, peak_idx, late_idx]
    titles = ['Early Universe', 'Peak Complexity', 'Late Universe']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, idx, title in zip(axes, indices, titles):
        pos = positions[idx]
        ax.scatter(pos[:, 0], pos[:, 1], s=15, c='blue', alpha=0.6)
        ax.set_title(f'{title}\nTime: {times[idx]:.0f}, Complexity: {complexities[idx]:.2f}')
        ax.set_aspect('equal')
        ax.set_facecolor('lightgray')
    
    plt.suptitle('Universe Snapshots: Three Eras', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Complete Exercise 2 first (complexity measures)")
    print("2. Fill in all TODOs")
    print("3. Run analyze_cosmic_evolution() for full analysis")
    print("4. Optional: Run visualize_universe() for animation")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # analyze_cosmic_evolution()
    
    # For animation:
    # universe = ToyUniverse(n_particles=200)
    # visualize_universe(universe)
