"""
Solution 3: Game of Life Complexity
===================================

Complete Game of Life with complexity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage
from solution_02_complexity_measure import compute_shannon_entropy


class GameOfLife:
    """Conway's Game of Life implementation."""
    
    def __init__(self, size=100):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.history = []
        self.generation = 0
        
    def random_soup(self, density=0.3):
        """Initialize with random cells."""
        self.grid = (np.random.random((self.size, self.size)) < density).astype(int)
        self.generation = 0
        
    def add_pattern(self, pattern, x, y):
        """Add pattern at position."""
        pattern = np.array(pattern)
        h, w = pattern.shape
        for i in range(h):
            for j in range(w):
                self.grid[(x+i) % self.size, (y+j) % self.size] = pattern[i, j]
    
    def count_neighbors(self):
        """Count live neighbors using convolution."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        return ndimage.convolve(self.grid, kernel, mode='wrap')
    
    def step(self):
        """One generation."""
        self.history.append(self.grid.copy())
        neighbors = self.count_neighbors()
        
        # Rules
        survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        birth = (self.grid == 0) & (neighbors == 3)
        
        self.grid = (survive | birth).astype(int)
        self.generation += 1
        
    def run(self, generations=100):
        for _ in range(generations):
            self.step()
    
    def get_population(self):
        return np.sum(self.grid)
    
    def get_density(self):
        return np.mean(self.grid)


PATTERNS = {
    'glider': [[0, 1, 0], [0, 0, 1], [1, 1, 1]],
    'blinker': [[1, 1, 1]],
    'block': [[1, 1], [1, 1]],
    'r_pentomino': [[0, 1, 1], [1, 1, 0], [0, 1, 0]]
}


def compute_gol_complexity(grid):
    """Compute complexity for binary Grid."""
    density = np.mean(grid)
    if density == 0 or density == 1:
        return {'entropy': 0, 'structure': 0, 'complexity': 0}
    
    binary_entropy = -density * np.log(density) - (1-density) * np.log(1-density)
    
    # Structure from edges
    h_edges = np.sum(np.abs(np.diff(grid, axis=1)))
    v_edges = np.sum(np.abs(np.diff(grid, axis=0)))
    structure = (h_edges + v_edges) / (grid.size * 2)
    
    complexity = binary_entropy * structure * 100
    
    return {
        'entropy': binary_entropy,
        'structure': structure,
        'complexity': complexity
    }


def track_gol_evolution(gol, generations=200):
    """Track GoL with complexity."""
    data = {k: [] for k in ['generations', 'population', 'density', 
                             'entropy', 'structure', 'complexity']}
    
    for gen in range(generations):
        metrics = compute_gol_complexity(gol.grid)
        data['generations'].append(gen)
        data['population'].append(gol.get_population())
        data['density'].append(gol.get_density())
        data['entropy'].append(metrics['entropy'])
        data['structure'].append(metrics['structure'])
        data['complexity'].append(metrics['complexity'])
        gol.step()
    
    return {k: np.array(v) for k, v in data.items()}


def compare_conditions():
    """Compare different initial conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = [
        ('Random 10%', 0.1),
        ('Random 30%', 0.3),
        ('Random 50%', 0.5),
        ('Random 70%', 0.7)
    ]
    
    for ax, (name, density) in zip(axes.flat, conditions):
        gol = GameOfLife(size=100)
        gol.random_soup(density)
        data = track_gol_evolution(gol, 300)
        
        ax.plot(data['generations'], data['complexity'], 'r-', lw=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Complexity')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        peak = data['generations'][np.argmax(data['complexity'])]
        ax.axvline(peak, color='orange', ls='--', alpha=0.7)
    
    plt.suptitle('Game of Life: Complexity vs Initial Density', fontweight='bold')
    plt.tight_layout()
    plt.show()


def demo():
    """Demo Game of Life."""
    print("Game of Life Demo")
    print("=" * 60)
    
    # Test blinker
    gol = GameOfLife(size=10)
    gol.add_pattern(PATTERNS['blinker'], 5, 4)
    init = gol.get_population()
    gol.step()
    step1 = gol.get_population()
    gol.step()
    step2 = gol.get_population()
    
    print(f"\n1. Blinker test: {init} -> {step1} -> {step2}")
    print(f"   Oscillates: {'✅' if init == step2 else '❌'}")
    
    # Random soup
    print("\n2. Random soup evolution:")
    gol2 = GameOfLife(100)
    gol2.random_soup(0.3)
    print(f"   Initial: {gol2.get_population()} cells")
    gol2.run(200)
    print(f"   After 200: {gol2.get_population()} cells")
    
    compare_conditions()


if __name__ == "__main__":
    demo()
