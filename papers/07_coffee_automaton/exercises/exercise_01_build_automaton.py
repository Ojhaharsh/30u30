"""
Exercise 1: Build the Coffee Automaton
=======================================

Goal: Implement a cellular automaton that models heat diffusion.

Your Task:
- Fill in the TODOs below to complete the implementation
- Run the simulation and visualize the results
- Observe how heat spreads from hot spots

Learning Objectives:
1. Understand how simple local rules create global behavior
2. Implement heat diffusion dynamics
3. Track system evolution over time
4. Measure when equilibrium is reached

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CoffeeAutomaton:
    """
    A cellular automaton simulating heat diffusion.
    
    Rules:
    - Each cell shares heat with its neighbors
    - Heat flows from hot to cold regions
    - Optional: cells lose heat to environment (cooling)
    """
    
    def __init__(self, size=50, diffusion_rate=0.2, cooling_rate=0.01):
        """
        Initialize the coffee automaton.
        
        Args:
            size: Grid size (size x size for 2D, or size for 1D)
            diffusion_rate: How fast heat spreads to neighbors (0-0.25)
            cooling_rate: How fast cells lose heat to environment (0 = no cooling)
        """
        self.size = size
        self.diffusion_rate = diffusion_rate
        self.cooling_rate = cooling_rate
        
        # TODO 1: Initialize the temperature grid
        # Start with zeros (cold) - we'll add hot spots later
        # Shape: (size, size) for 2D grid
        self.grid = None  # TODO: Initialize with np.zeros
        
        # Track history for analysis
        self.history = []
        
    def add_hotspot(self, x, y, temperature=100.0, radius=3):
        """
        Add a hot spot to the grid.
        
        Args:
            x, y: Center coordinates of hotspot
            temperature: Temperature value at center
            radius: Size of the hotspot
        """
        # TODO 2: Add a circular hot spot to the grid
        # Hint: Create a circular mask and set those cells to high temperature
        # Use: np.sqrt((i - x)**2 + (j - y)**2) <= radius
        pass  # TODO: Implement
    
    def step(self):
        """
        Perform one time step of the simulation.
        
        Heat diffusion rule:
        new_temp[i,j] = old_temp[i,j] + diffusion_rate * (neighbor_avg - old_temp[i,j])
        
        Then apply cooling:
        final_temp[i,j] = new_temp[i,j] * (1 - cooling_rate)
        """
        # Save current state to history
        self.history.append(self.grid.copy())
        
        # TODO 3: Compute the average of neighbors for each cell
        # For 2D: each cell has up to 4 neighbors (von Neumann) or 8 (Moore)
        # Use np.roll for periodic boundaries, or handle edges specially
        # Hint: neighbor_sum = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + ...
        
        neighbor_sum = None  # TODO: Sum of all neighbors
        neighbor_count = None  # TODO: Number of neighbors (handle edges)
        neighbor_avg = None  # TODO: neighbor_sum / neighbor_count
        
        # TODO 4: Apply diffusion rule
        # new_grid = old_grid + diffusion_rate * (neighbor_avg - old_grid)
        new_grid = None  # TODO: Implement diffusion
        
        # TODO 5: Apply cooling (heat loss to environment)
        # new_grid = new_grid * (1 - cooling_rate)
        new_grid = None  # TODO: Apply cooling
        
        # TODO 6: Update the grid
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
        """Return variance of temperatures (measure of non-uniformity)."""
        return np.var(self.grid)
    
    def is_equilibrium(self, threshold=0.01):
        """Check if system has reached equilibrium (uniform temperature)."""
        return self.get_variance() < threshold


class CoffeeAutomaton1D:
    """
    A 1D version (simpler to start with).
    
    Imagine a line of coffee cups - heat spreads left and right.
    """
    
    def __init__(self, size=100, diffusion_rate=0.2, cooling_rate=0.01):
        self.size = size
        self.diffusion_rate = diffusion_rate
        self.cooling_rate = cooling_rate
        
        # TODO 7: Initialize 1D temperature array
        self.grid = None  # TODO: np.zeros(size)
        self.history = []
        
    def add_hotspot(self, position, temperature=100.0, width=5):
        """Add a hot region to the 1D grid."""
        # TODO 8: Set cells in range [position-width, position+width] 
        # to high temperature (with falloff from center)
        pass  # TODO: Implement
        
    def step(self):
        """One step of 1D diffusion."""
        self.history.append(self.grid.copy())
        
        # TODO 9: 1D diffusion - each cell averages with left and right neighbors
        # left_neighbor = np.roll(grid, 1)
        # right_neighbor = np.roll(grid, -1)
        # neighbor_avg = (left_neighbor + right_neighbor) / 2
        
        left = None  # TODO
        right = None  # TODO
        neighbor_avg = None  # TODO
        
        # Apply diffusion
        new_grid = None  # TODO
        
        # Apply cooling
        new_grid = None  # TODO
        
        self.grid = new_grid
        
    def run(self, steps=100):
        """Run for given steps."""
        for _ in range(steps):
            self.step()


def visualize_2d(automaton, interval=100):
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
    
    ani = FuncAnimation(fig, update, frames=200, interval=interval, blit=True)
    plt.show()
    return ani


def visualize_1d_evolution(automaton, steps=200):
    """Show 1D evolution as a 2D heatmap (x = position, y = time)."""
    # Run simulation
    for _ in range(steps):
        automaton.step()
    
    # Plot history as heatmap
    history_array = np.array(automaton.history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    im = ax1.imshow(history_array, aspect='auto', cmap='hot', origin='lower')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Time Step')
    ax1.set_title('1D Coffee Automaton Evolution')
    plt.colorbar(im, ax=ax1, label='Temperature')
    
    # Total heat over time
    total_heat = [np.sum(h) for h in automaton.history]
    ax2.plot(total_heat, 'b-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Total Heat')
    ax2.set_title('Heat Conservation (with cooling)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_implementation():
    """Test your implementation."""
    print("Testing Coffee Automaton Implementation...")
    print("=" * 60)
    
    # Test 1D
    print("\n1. Testing 1D Automaton:")
    ca1d = CoffeeAutomaton1D(size=100)
    ca1d.add_hotspot(50, temperature=100)
    
    initial_heat = ca1d.get_total_heat() if hasattr(ca1d, 'get_total_heat') else np.sum(ca1d.grid)
    print(f"   Initial total heat: {initial_heat:.1f}")
    
    ca1d.run(steps=50)
    final_heat = ca1d.get_total_heat() if hasattr(ca1d, 'get_total_heat') else np.sum(ca1d.grid)
    print(f"   Final total heat: {final_heat:.1f}")
    print(f"   Heat lost: {initial_heat - final_heat:.1f} ({(initial_heat - final_heat) / initial_heat * 100:.1f}%)")
    
    # Test 2D
    print("\n2. Testing 2D Automaton:")
    ca2d = CoffeeAutomaton(size=50)
    ca2d.add_hotspot(25, 25, temperature=100)
    
    initial_variance = ca2d.get_variance() if hasattr(ca2d, 'get_variance') else np.var(ca2d.grid)
    print(f"   Initial variance: {initial_variance:.2f}")
    
    ca2d.run(steps=100)
    final_variance = ca2d.get_variance() if hasattr(ca2d, 'get_variance') else np.var(ca2d.grid)
    print(f"   Final variance: {final_variance:.2f}")
    print(f"   Is equilibrium: {ca2d.is_equilibrium() if hasattr(ca2d, 'is_equilibrium') else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("✅ If heat spreads from hotspots and variance decreases,")
    print("   your implementation is likely correct!")
    print("=" * 60)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs (search for 'TODO')")
    print("2. Run this file to test your implementation")
    print("3. Heat should spread from hotspots")
    print("4. Variance should decrease (approaching equilibrium)")
    print("5. Check solutions/solution_01_build_automaton.py if stuck")
    print("=" * 60 + "\n")
    
    # Uncomment when ready to test:
    # test_implementation()
    
    # Uncomment to visualize 1D:
    # ca1d = CoffeeAutomaton1D(size=100)
    # ca1d.add_hotspot(50, temperature=100)
    # visualize_1d_evolution(ca1d)
    
    # Uncomment to visualize 2D:
    # ca2d = CoffeeAutomaton(size=50)
    # ca2d.add_hotspot(25, 25, temperature=100)
    # visualize_2d(ca2d)
