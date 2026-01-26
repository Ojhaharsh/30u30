"""
Solution 5: Cosmological Model
==============================

Toy universe simulation showing complexity peak during structure formation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def compute_shannon_entropy(x, num_bins=20):
    flat = x.flatten()
    if np.max(flat) == np.min(flat):
        return 0.0
    hist, _ = np.histogram(flat, bins=num_bins)
    p = hist / np.sum(hist)
    return -np.sum(p[p > 0] * np.log(p[p > 0]))


class ToyUniverse:
    """Toy cosmic evolution model."""
    
    def __init__(self, n_particles=200, box_size=100.0,
                 gravity_strength=0.1, thermal_strength=0.5,
                 expansion_rate=0.001):
        self.n_particles = n_particles
        self.box_size = box_size
        self.gravity_strength = gravity_strength
        self.thermal_strength = thermal_strength
        self.expansion_rate = expansion_rate
        
        # Initialize
        self.positions = np.random.uniform(0, box_size, (n_particles, 2))
        self.velocities = np.random.randn(n_particles, 2) * thermal_strength
        
        self.time = 0.0
        self.scale_factor = 1.0
        
        self.history = {k: [] for k in [
            'time', 'scale_factor', 'positions',
            'complexity', 'clustering', 'entropy'
        ]}
        
    def compute_gravity(self):
        """Gravitational accelerations."""
        acc = np.zeros_like(self.positions)
        
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i == j:
                    continue
                r = self.positions[j] - self.positions[i]
                dist = np.sqrt(np.sum(r**2) + 1.0)
                acc[i] += self.gravity_strength * r / (dist**3)
        
        return acc
    
    def compute_thermal_kick(self):
        """Random thermal motion."""
        temp = self.thermal_strength / (self.scale_factor ** 0.5)
        return np.random.randn(self.n_particles, 2) * temp
    
    def step(self, dt=1.0):
        """Evolve one step."""
        self.time += dt
        
        old_scale = self.scale_factor
        self.scale_factor = 1.0 + self.expansion_rate * self.time
        
        # Hubble drag
        self.velocities *= old_scale / self.scale_factor
        
        # Gravity
        self.velocities += self.compute_gravity() * dt
        
        # Thermal
        self.velocities += self.compute_thermal_kick() * np.sqrt(dt)
        
        # Update positions
        self.positions += self.velocities * dt
        
        # Periodic BC
        current_box = self.box_size * self.scale_factor
        self.positions = self.positions % current_box
        
        self._record()
    
    def _record(self):
        self.history['time'].append(self.time)
        self.history['scale_factor'].append(self.scale_factor)
        self.history['positions'].append(self.positions.copy())
        
        c = self.compute_complexity()
        self.history['complexity'].append(c['total'])
        self.history['clustering'].append(c['clustering'])
        self.history['entropy'].append(c['entropy'])
    
    def compute_complexity(self):
        """Complexity of particle distribution."""
        n_cells = 10
        current_box = self.box_size * self.scale_factor
        cell_size = current_box / n_cells
        
        counts = np.zeros((n_cells, n_cells))
        for pos in self.positions:
            i = int(pos[0] / cell_size) % n_cells
            j = int(pos[1] / cell_size) % n_cells
            counts[i, j] += 1
        
        clustering = np.var(counts) / (np.mean(counts) + 1e-10)
        
        entropy = compute_shannon_entropy(counts)
        max_entropy = np.log(n_cells * n_cells)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        complexity = clustering * norm_entropy * (1 - norm_entropy) * 4
        
        return {
            'clustering': clustering,
            'entropy': norm_entropy,
            'total': complexity if np.isfinite(complexity) else 0
        }
    
    def run(self, steps=500, dt=1.0):
        for _ in range(steps):
            self.step(dt)


def analyze_evolution():
    """Full analysis."""
    print("Cosmic Evolution Simulation")
    print("=" * 60)
    
    universe = ToyUniverse(
        n_particles=300,
        box_size=100.0,
        gravity_strength=0.05,
        thermal_strength=0.3,
        expansion_rate=0.002
    )
    
    universe.run(1000, 1.0)
    
    times = np.array(universe.history['time'])
    complexities = np.array(universe.history['complexity'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scale factor
    axes[0, 0].plot(times, universe.history['scale_factor'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Cosmic Time')
    axes[0, 0].set_ylabel('Scale Factor')
    axes[0, 0].set_title('Universe Expansion')
    axes[0, 0].grid(alpha=0.3)
    
    # Clustering
    axes[0, 1].plot(times, universe.history['clustering'], 'g-', lw=2)
    axes[0, 1].set_xlabel('Cosmic Time')
    axes[0, 1].set_ylabel('Clustering')
    axes[0, 1].set_title('Structure Formation')
    axes[0, 1].grid(alpha=0.3)
    
    # Entropy
    axes[1, 0].plot(times, universe.history['entropy'], 'm-', lw=2)
    axes[1, 0].set_xlabel('Cosmic Time')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Entropy Evolution')
    axes[1, 0].grid(alpha=0.3)
    
    # Complexity
    ax = axes[1, 1]
    ax.plot(times, complexities, 'r-', lw=2)
    ax.set_xlabel('Cosmic Time')
    ax.set_ylabel('Complexity')
    ax.set_title('Cosmic Complexity')
    ax.grid(alpha=0.3)
    
    peak_idx = np.argmax(complexities)
    peak_time = times[peak_idx]
    ax.axvline(peak_time, color='orange', ls='--', label=f'Peak t={peak_time:.0f}')
    ax.legend()
    
    plt.suptitle('Toy Cosmology: Complexity Evolution', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Snapshots
    show_snapshots(universe)
    
    print(f"\n🎯 Peak complexity at t={peak_time:.0f}")
    print("This is when complex life could exist!")


def show_snapshots(universe):
    """Show universe at different times."""
    times = np.array(universe.history['time'])
    positions = universe.history['positions']
    complexities = np.array(universe.history['complexity'])
    
    early = len(times) // 10
    peak = np.argmax(complexities)
    late = len(times) - 1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, idx, title in zip(axes, [early, peak, late], 
                               ['Early', 'Peak', 'Late']):
        ax.scatter(positions[idx][:, 0], positions[idx][:, 1], 
                  s=10, alpha=0.6)
        ax.set_title(f'{title}: t={times[idx]:.0f}, C={complexities[idx]:.2f}')
        ax.set_aspect('equal')
    
    plt.suptitle('Universe Snapshots', fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_evolution()
