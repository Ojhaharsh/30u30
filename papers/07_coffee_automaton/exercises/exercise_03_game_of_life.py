"""
Exercise 3: Game of Life Complexity
===================================

Goal: Apply complexity measures to Conway's Game of Life and observe 
the rise-and-fall pattern of complexity.

Your Task:
- Implement the Game of Life rules
- Apply complexity measures from Exercise 2
- Analyze different initial conditions
- Identify phases: chaos → structure → stability

Learning Objectives:
1. Understand discrete cellular automata rules
2. Compare continuous (coffee) vs discrete (life) systems
3. See how structures emerge from chaos
4. Classify patterns (still lifes, oscillators, spaceships)

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage

# Import complexity measures from Exercise 2
try:
    from exercise_02_complexity_measure import (
        compute_shannon_entropy, 
        compute_gradient_structure,
        compute_complexity
    )
except ImportError:
    print("Warning: Complete Exercise 2 first for complexity functions!")


class GameOfLife:
    """
    Conway's Game of Life implementation.
    
    Rules:
    - Any live cell with 2 or 3 live neighbors survives
    - Any dead cell with exactly 3 live neighbors becomes alive
    - All other cells die or stay dead
    """
    
    def __init__(self, size=100):
        """
        Initialize empty grid.
        
        Args:
            size: Grid size (size x size)
        """
        self.size = size
        # TODO 1: Initialize grid with zeros (all dead)
        self.grid = None  # TODO: np.zeros((size, size), dtype=int)
        self.history = []
        self.generation = 0
        
    def random_soup(self, density=0.3):
        """
        Initialize with random live cells.
        
        Args:
            density: Fraction of cells that are alive (0-1)
        """
        # TODO 2: Create random initial state
        # Each cell has 'density' probability of being alive
        self.grid = None  # TODO: (np.random.random((size, size)) < density).astype(int)
        self.generation = 0
        
    def add_pattern(self, pattern, x, y):
        """
        Add a pre-defined pattern at position (x, y).
        
        Args:
            pattern: 2D array of 0s and 1s
            x, y: Top-left corner position
        """
        pattern = np.array(pattern)
        h, w = pattern.shape
        
        # TODO 3: Place pattern on grid (handle boundaries)
        # Hint: Use modulo for wrap-around
        for i in range(h):
            for j in range(w):
                pass  # TODO: self.grid[(x+i) % self.size, (y+j) % self.size] = pattern[i, j]
    
    def count_neighbors(self):
        """
        Count the number of live neighbors for each cell.
        
        Returns:
            2D array of neighbor counts
        """
        # TODO 4: Count neighbors using convolution
        # Kernel: 3x3 of all 1s except center is 0
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Use scipy.ndimage.convolve with wrap mode for periodic boundaries
        neighbor_count = None  # TODO: ndimage.convolve(self.grid, kernel, mode='wrap')
        
        return neighbor_count
    
    def step(self):
        """Perform one generation step."""
        self.history.append(self.grid.copy())
        
        # TODO 5: Count neighbors
        neighbors = None  # TODO: self.count_neighbors()
        
        # TODO 6: Apply rules
        # Rule 1: Live cell with 2 or 3 neighbors survives
        # Rule 2: Dead cell with exactly 3 neighbors becomes alive
        # Rule 3: All others die
        
        # Boolean array approach:
        # survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        # birth = (self.grid == 0) & (neighbors == 3)
        # new_grid = (survive | birth).astype(int)
        
        new_grid = None  # TODO: Implement rules
        
        self.grid = new_grid
        self.generation += 1
        
    def run(self, generations=100):
        """Run for given number of generations."""
        for _ in range(generations):
            self.step()
    
    def get_population(self):
        """Return number of live cells."""
        return np.sum(self.grid)
    
    def get_density(self):
        """Return fraction of live cells."""
        return np.mean(self.grid)


# Pre-defined patterns
PATTERNS = {
    'glider': [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ],
    'blinker': [
        [1, 1, 1]
    ],
    'block': [
        [1, 1],
        [1, 1]
    ],
    'beehive': [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ],
    'r_pentomino': [
        [0, 1, 1],
        [1, 1, 0],
        [0, 1, 0]
    ],
    'glider_gun': [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
}


def compute_gol_complexity(grid):
    """
    Compute complexity for Game of Life (binary grid).
    
    For binary grids, we need slightly different measures.
    """
    # TODO 7: Adapt complexity measure for binary grid
    # Option 1: Use entropy on local patterns (2x2 or 3x3 windows)
    # Option 2: Use spatial entropy based on block patterns
    # Option 3: Simple entropy on population fraction per region
    
    # Simple approach: entropy based on density + structure
    density = np.mean(grid)
    binary_entropy = -density * np.log(density + 1e-10) - (1-density) * np.log(1-density + 1e-10)
    
    # Structure: Count pattern boundaries
    # An edge exists between cells of different states
    h_edges = np.sum(np.abs(np.diff(grid, axis=1)))
    v_edges = np.sum(np.abs(np.diff(grid, axis=0)))
    structure = (h_edges + v_edges) / (grid.size * 2)
    
    complexity = binary_entropy * structure * 100  # Scale for visibility
    
    return {
        'entropy': binary_entropy if np.isfinite(binary_entropy) else 0,
        'structure': structure,
        'complexity': complexity if np.isfinite(complexity) else 0
    }


def track_gol_evolution(gol, generations=200):
    """
    Track Game of Life evolution with complexity measures.
    
    Args:
        gol: GameOfLife instance
        generations: Number of generations to run
        
    Returns:
        Dict with evolution data
    """
    data = {
        'generations': [],
        'population': [],
        'density': [],
        'entropy': [],
        'structure': [],
        'complexity': []
    }
    
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


def compare_initial_conditions():
    """
    Compare complexity evolution for different initial conditions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    conditions = [
        ('Random Soup (10%)', lambda g: g.random_soup(0.1)),
        ('Random Soup (30%)', lambda g: g.random_soup(0.3)),
        ('Random Soup (50%)', lambda g: g.random_soup(0.5)),
        ('R-pentomino', lambda g: (g.random_soup(0.0) or True) and g.add_pattern(PATTERNS['r_pentomino'], 50, 50)),
        ('Glider Gun', lambda g: (g.random_soup(0.0) or True) and g.add_pattern(PATTERNS['glider_gun'], 10, 10)),
        ('Single Glider', lambda g: (g.random_soup(0.0) or True) and g.add_pattern(PATTERNS['glider'], 10, 10))
    ]
    
    for idx, (name, init_func) in enumerate(conditions):
        ax = axes[idx // 3, idx % 3]
        
        gol = GameOfLife(size=100)
        init_func(gol)
        
        data = track_gol_evolution(gol, generations=300)
        
        ax.plot(data['generations'], data['complexity'], 'r-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Complexity')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        # Find peak
        peak_gen = data['generations'][np.argmax(data['complexity'])]
        ax.axvline(x=peak_gen, color='orange', linestyle='--', alpha=0.7)
    
    plt.suptitle('Game of Life Complexity: Different Initial Conditions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_gol(gol, generations=200, interval=50):
    """Create animation of Game of Life."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im = ax1.imshow(gol.grid, cmap='binary')
    ax1.set_title(f'Generation 0 | Population: {gol.get_population()}')
    
    complexities = []
    
    line, = ax2.plot([], [], 'r-', linewidth=2)
    ax2.set_xlim(0, generations)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Complexity')
    ax2.set_title('Complexity Over Time')
    ax2.grid(True, alpha=0.3)
    
    def update(frame):
        gol.step()
        im.set_array(gol.grid)
        ax1.set_title(f'Generation {gol.generation} | Population: {gol.get_population()}')
        
        metrics = compute_gol_complexity(gol.grid)
        complexities.append(metrics['complexity'])
        
        line.set_data(range(len(complexities)), complexities)
        ax2.set_ylim(0, max(complexities) * 1.1 + 0.1)
        
        return [im, line]
    
    ani = FuncAnimation(fig, update, frames=generations, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


def test_game_of_life():
    """Test Game of Life implementation."""
    print("Testing Game of Life...")
    print("=" * 60)
    
    # Test rules
    gol = GameOfLife(size=10)
    
    # Test blinker oscillator
    gol.add_pattern(PATTERNS['blinker'], 5, 4)
    initial_pop = gol.get_population()
    print(f"\n1. Blinker test:")
    print(f"   Initial population: {initial_pop}")
    
    gol.step()
    step1_pop = gol.get_population()
    print(f"   After 1 step: {step1_pop}")
    
    gol.step()
    step2_pop = gol.get_population()
    print(f"   After 2 steps (should match initial): {step2_pop}")
    
    if initial_pop == step2_pop:
        print("   ✅ Blinker oscillates correctly!")
    else:
        print("   ❌ Blinker test failed")
    
    # Test random soup
    print("\n2. Random soup evolution:")
    gol2 = GameOfLife(size=50)
    gol2.random_soup(0.3)
    
    initial = gol2.get_population()
    print(f"   Initial population: {initial}")
    
    gol2.run(100)
    final = gol2.get_population()
    print(f"   After 100 generations: {final}")
    
    print("\n" + "=" * 60)
    print("✅ If blinker oscillates and population stabilizes,")
    print("   your implementation is correct!")
    print("=" * 60)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Complete Exercises 1 and 2 first")
    print("2. Fill in all TODOs in this file")
    print("3. Run test_game_of_life() to verify")
    print("4. Run compare_initial_conditions() for analysis")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # test_game_of_life()
    # compare_initial_conditions()
    
    # Visualize random soup:
    # gol = GameOfLife(size=100)
    # gol.random_soup(0.3)
    # visualize_gol(gol)
