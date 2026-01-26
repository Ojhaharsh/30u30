"""
Solution 1: Build the Coffee Automaton
======================================

Complete implementation of 1D and 2D coffee automaton (heat diffusion).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CoffeeAutomaton:
    """
    A cellular automaton simulating heat diffusion in 2D.
    """
    
    def __init__(self, size=50, diffusion_rate=0.2, cooling_rate=0.01):
        self.size = size
        self.diffusion_rate = diffusion_rate
        self.cooling_rate = cooling_rate
        
        # Initialize temperature grid (all cold)
        self.grid = np.zeros((size, size))
        
        # Track history for analysis
        self.history = []
        
    def add_hotspot(self, x, y, temperature=100.0, radius=3):
        """Add a circular hot spot to the grid."""
        for i in range(self.size):
            for j in range(self.size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist <= radius:
                    # Temperature falls off with distance
                    falloff = 1.0 - (dist / radius)
                    self.grid[i, j] = max(self.grid[i, j], temperature * falloff)
    
    def step(self):
        """Perform one time step of the simulation."""
        self.history.append(self.grid.copy())
        
        # Compute neighbor sum using roll (periodic boundaries)
        neighbor_sum = (
            np.roll(self.grid, 1, axis=0) +   # Up
            np.roll(self.grid, -1, axis=0) +  # Down
            np.roll(self.grid, 1, axis=1) +   # Left
            np.roll(self.grid, -1, axis=1)    # Right
        )
        neighbor_avg = neighbor_sum / 4.0
        
        # Apply diffusion rule
        new_grid = self.grid + self.diffusion_rate * (neighbor_avg - self.grid)
        
        # Apply cooling
        new_grid = new_grid * (1 - self.cooling_rate)
        
        # Update grid
        self.grid = new_grid
        
    def run(self, steps=100):
        """Run simulation for given number of steps."""
        for _ in range(steps):
            self.step()
    
    def get_total_heat(self):
        """Return total heat in the system."""
        return np.sum(self.grid)
    
    def get_max_temperature(self):
        """Return maximum temperature."""
        return np.max(self.grid)
    
    def get_variance(self):
        """Return variance of temperatures."""
        return np.var(self.grid)
    
    def is_equilibrium(self, threshold=0.01):
        """Check if system has reached equilibrium."""
        return self.get_variance() < threshold


class CoffeeAutomaton1D:
    """
    A 1D version of the coffee automaton.
    """
    
    def __init__(self, size=100, diffusion_rate=0.2, cooling_rate=0.01):
        self.size = size
        self.diffusion_rate = diffusion_rate
        self.cooling_rate = cooling_rate
        
        # Initialize 1D temperature array
        self.grid = np.zeros(size)
        self.history = []
        
    def add_hotspot(self, position, temperature=100.0, width=5):
        """Add a hot region to the 1D grid."""
        for i in range(self.size):
            dist = abs(i - position)
            if dist <= width:
                falloff = 1.0 - (dist / width)
                self.grid[i] = max(self.grid[i], temperature * falloff)
        
    def step(self):
        """One step of 1D diffusion."""
        self.history.append(self.grid.copy())
        
        # 1D diffusion
        left = np.roll(self.grid, 1)
        right = np.roll(self.grid, -1)
        neighbor_avg = (left + right) / 2
        
        # Apply diffusion
        new_grid = self.grid + self.diffusion_rate * (neighbor_avg - self.grid)
        
        # Apply cooling
        new_grid = new_grid * (1 - self.cooling_rate)
        
        self.grid = new_grid
        
    def run(self, steps=100):
        """Run for given steps."""
        for _ in range(steps):
            self.step()
    
    def get_total_heat(self):
        return np.sum(self.grid)
    
    def get_variance(self):
        return np.var(self.grid)


def visualize_2d(automaton, steps=200, interval=50):
    """Create animation of 2D automaton."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(automaton.grid, cmap='hot', vmin=0, vmax=100)
    plt.colorbar(im, label='Temperature')
    ax.set_title('Coffee Automaton - Heat Diffusion')
    
    def update(frame):
        automaton.step()
        im.set_array(automaton.grid)
        ax.set_title(f'Step {frame} | Total Heat: {automaton.get_total_heat():.1f}')
        return [im]
    
    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    plt.show()
    return ani


def visualize_1d_evolution(automaton, steps=200):
    """Show 1D evolution as a 2D heatmap."""
    for _ in range(steps):
        automaton.step()
    
    history_array = np.array(automaton.history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    im = ax1.imshow(history_array, aspect='auto', cmap='hot', origin='lower')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Time Step')
    ax1.set_title('1D Coffee Automaton Evolution')
    plt.colorbar(im, ax=ax1, label='Temperature')
    
    total_heat = [np.sum(h) for h in automaton.history]
    ax2.plot(total_heat, 'b-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Total Heat')
    ax2.set_title('Heat Conservation (with cooling)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo():
    """Demonstrate the coffee automaton."""
    print("Coffee Automaton Demo")
    print("=" * 60)
    
    # 1D Demo
    print("\n1. 1D Coffee Automaton:")
    ca1d = CoffeeAutomaton1D(size=100, diffusion_rate=0.2, cooling_rate=0.01)
    ca1d.add_hotspot(50, temperature=100, width=5)
    
    print(f"   Initial heat: {ca1d.get_total_heat():.1f}")
    ca1d.run(100)
    print(f"   Final heat: {ca1d.get_total_heat():.1f}")
    print(f"   Final variance: {ca1d.get_variance():.4f}")
    
    # 2D Demo
    print("\n2. 2D Coffee Automaton:")
    ca2d = CoffeeAutomaton(size=50, diffusion_rate=0.15, cooling_rate=0.005)
    ca2d.add_hotspot(25, 25, temperature=100, radius=5)
    
    print(f"   Initial heat: {ca2d.get_total_heat():.1f}")
    print(f"   Initial variance: {ca2d.get_variance():.2f}")
    
    ca2d.run(200)
    
    print(f"   Final heat: {ca2d.get_total_heat():.1f}")
    print(f"   Final variance: {ca2d.get_variance():.4f}")
    print(f"   Equilibrium: {ca2d.is_equilibrium()}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete! Try the visualization functions.")
    

if __name__ == "__main__":
    demo()
    
    # Visualize 1D
    print("\nVisualizing 1D evolution...")
    ca1d = CoffeeAutomaton1D(size=100)
    ca1d.add_hotspot(50, temperature=100)
    visualize_1d_evolution(ca1d, steps=200)
    
    # Visualize 2D (uncomment for animation)
    # print("\nVisualizing 2D evolution...")
    # ca2d = CoffeeAutomaton(size=50)
    # ca2d.add_hotspot(25, 25, temperature=100)
    # visualize_2d(ca2d)
