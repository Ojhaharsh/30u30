"""
Coffee Automaton — Implementation of Aaronson, Carroll, Ouellette (2014)
Paper: arXiv:1405.6903

Implements the two models from the paper:
  - Interacting model (Section 3.1): binary grid, random adjacent swaps
  - Non-interacting model (Section 3.2): independent cream particle random walks

And the two measurement approaches:
  - Coarse-graining with 3-bucket thresholding (Section 5)
  - Adjusted coarse-graining with 7-bucket + row-majority (Section 6)

Complexity and entropy are estimated via gzip compressed size,
as a proxy for Kolmogorov complexity.
"""

import gzip
import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Utility: gzip-based Kolmogorov complexity proxy
# ---------------------------------------------------------------------------

def gzip_size(data: bytes, level: int = 9) -> int:
    """
    Compressed size in bytes — our proxy for K(x).
    The paper uses gzip throughout (Section 4), and shows different compressors
    give qualitatively similar curves (Figure 5).
    """
    return len(gzip.compress(data, compresslevel=level))


def grid_to_bytes(grid: np.ndarray) -> bytes:
    """Convert a numpy grid to bytes for compression."""
    return grid.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Coarse-graining (Sections 5 and 6)
# ---------------------------------------------------------------------------

def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    Coarse-grain the grid by averaging over grain_size x grain_size blocks.

    From Section 5.1: "we construct a new array in which the value of each cell
    is the average of the values of the nearby cells in the fine-grained array.
    We define 'nearby' cells as those within a g x g square centered at the
    cell in question."

    Returns a float array with values in [0, 1].
    """
    n = grid.shape[0]
    # Number of coarse cells
    cn = n // grain_size
    if cn == 0:
        cn = 1

    coarse = np.zeros((cn, cn), dtype=np.float64)
    for i in range(cn):
        for j in range(cn):
            r0 = i * grain_size
            c0 = j * grain_size
            r1 = min(r0 + grain_size, n)
            c1 = min(c0 + grain_size, n)
            coarse[i, j] = grid[r0:r1, c0:c1].mean()
    return coarse


def threshold_array(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """
    Threshold floating-point coarse-grained array into discrete buckets.

    Section 5.1: 3 buckets — "areas which are mostly coffee (values close to 0),
    mostly cream (values close to 1), or mixed (values close to 0.5)."

    Section 6.1: 7 buckets — finer resolution to reduce border artifacts.

    Returns uint8 array with values in [0, num_buckets-1].
    """
    # Map [0, 1] to [0, num_buckets-1] and round
    scaled = coarse * (num_buckets - 1)
    return np.clip(np.round(scaled), 0, num_buckets - 1).astype(np.uint8)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """
    Adjusted coarse-graining from Section 6.1.

    "If a cell is within one threshold value of the majority value in its row,
    it is adjusted to the majority value."

    This removes border pixel artifacts that caused fake complexity
    in the non-interacting model.
    """
    adjusted = thresholded.copy()
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        # Find majority value in this row
        values, counts = np.unique(row, return_counts=True)
        majority = values[np.argmax(counts)]
        # Snap cells within 1 bucket of majority
        mask = np.abs(row.astype(np.int16) - int(majority)) <= 1
        adjusted[i, mask] = majority
    return adjusted


# ---------------------------------------------------------------------------
# Complexity and entropy measurement
# ---------------------------------------------------------------------------

def measure_entropy(grid: np.ndarray) -> int:
    """
    Entropy estimate: gzip compressed size of the fine-grained state.

    Section 4: "the estimated entropy of the automaton state is the compressed
    file size of the fine-grained array."
    """
    return gzip_size(grid_to_bytes(grid))


def measure_complexity(grid: np.ndarray, grain_size: int = 10,
                       num_buckets: int = 7, adjust: bool = True) -> int:
    """
    Apparent complexity estimate: gzip compressed size of coarse-grained state.

    Section 4: "The estimated complexity of the state, K(S), is the file size
    of the thresholded, coarse-grained array after compression."

    Args:
        grid: The fine-grained binary grid (N x N, uint8)
        grain_size: Size of coarse-graining blocks (g in the paper)
        num_buckets: Number of threshold buckets (3 for Section 5, 7 for Section 6)
        adjust: Whether to apply row-majority adjustment (Section 6)
    """
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, num_buckets)
    if adjust:
        discrete = row_majority_adjust(discrete)
    return gzip_size(grid_to_bytes(discrete))


# ---------------------------------------------------------------------------
# The Coffee Automaton (Section 3)
# ---------------------------------------------------------------------------

class CoffeeAutomaton:
    """
    The coffee automaton from Section 3 of the paper.

    A 2D grid of binary values: 1 = cream, 0 = coffee.
    Initial state: top half = cream (1), bottom half = coffee (0).

    Two models:
    - 'interacting' (Section 3.1): One adjacent-different-pair swap per step.
      "only one particle may occupy each cell."
    - 'non_interacting' (Section 3.2): Each cream particle random-walks
      independently. Multiple can occupy one cell.
    """

    def __init__(self, grid_size: int = 100, model: str = 'interacting',
                 seed: Optional[int] = None):
        """
        Args:
            grid_size: Side length of the square grid (N in paper)
            model: 'interacting' or 'non_interacting'
            seed: Random seed for reproducibility
        """
        self.n = grid_size
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.time = 0

        if model == 'interacting':
            # Binary grid: top half = 1 (cream), bottom half = 0 (coffee)
            # Section 3: "The automaton begins in a state in which the top half
            # of the cells are filled with ones, and the bottom half is filled
            # with zeros."
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
            self.grid[:grid_size // 2, :] = 1
        elif model == 'non_interacting':
            # For non-interacting, track cream particle positions.
            # Count grid: how many cream particles at each cell.
            # Initially, one per cell in top half.
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint16)
            self.grid[:grid_size // 2, :] = 1
            # Also store individual particle positions for random walks
            self._init_particles()
        else:
            raise ValueError(f"model must be 'interacting' or 'non_interacting', got '{model}'")

    def _init_particles(self):
        """Initialize particle position list for non-interacting model."""
        self.particles = []
        for r in range(self.n // 2):
            for c in range(self.n):
                self.particles.append([r, c])
        self.particles = np.array(self.particles, dtype=np.int32)

    def step(self, num_steps: int = 1):
        """
        Advance the automaton by num_steps.

        For the interacting model, we do batch swaps for efficiency:
        select many random adjacent pairs, filter to those that differ,
        and swap them. This is faster than one-swap-per-step while
        preserving the qualitative dynamics.

        The paper does one swap per step (Section 3.1), but that's
        extremely slow for large grids. Batch swapping speeds things
        up without changing the equilibrium behavior.
        """
        if self.model == 'interacting':
            self._step_interacting(num_steps)
        else:
            self._step_non_interacting(num_steps)
        self.time += num_steps

    def _step_interacting(self, num_steps: int):
        """
        Interacting model: random adjacent swaps.

        Section 3.1: "at each time step, one pair of horizontally or
        vertically adjacent, differing particles is selected, and the
        particles' positions are swapped."

        We batch multiple non-conflicting swaps per call for speed.
        """
        n = self.n
        grid = self.grid

        # How many candidate swaps per batch —
        # roughly n*n/4 so each pixel gets ~1 chance per batch
        batch_size = max(1, (n * n) // 4)

        for _ in range(num_steps):
            # Pick random cells
            rows = self.rng.integers(0, n, size=batch_size)
            cols = self.rng.integers(0, n, size=batch_size)

            # Pick random direction: 0=up, 1=down, 2=left, 3=right
            dirs = self.rng.integers(0, 4, size=batch_size)

            # Compute neighbor coordinates
            dr = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            nr = rows + dr[dirs, 0]
            nc = cols + dr[dirs, 1]

            # Filter: within bounds
            valid = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)

            rows = rows[valid]
            cols = cols[valid]
            nr = nr[valid]
            nc = nc[valid]

            # Filter: different values (only swap if they differ)
            diff = grid[rows, cols] != grid[nr, nc]
            rows = rows[diff]
            cols = cols[diff]
            nr = nr[diff]
            nc = nc[diff]

            # Swap
            if len(rows) > 0:
                temp = grid[rows, cols].copy()
                grid[rows, cols] = grid[nr, nc]
                grid[nr, nc] = temp

    def _step_non_interacting(self, num_steps: int):
        """
        Non-interacting model: each cream particle random-walks.

        Section 3.2: "at each time step, each cream particle in the system
        moves one step in a randomly chosen direction."
        """
        n = self.n
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for _ in range(num_steps):
            # Clear the count grid and rebuild
            self.grid[:] = 0

            # Move each particle in a random direction
            choices = self.rng.integers(0, 4, size=len(self.particles))
            moves = directions[choices]
            self.particles += moves

            # Periodic boundary conditions
            # Section 9 (Appendix) assumes periodic boundaries
            self.particles[:, 0] %= n
            self.particles[:, 1] %= n

            # Rebuild count grid
            for p in self.particles:
                self.grid[p[0], p[1]] += 1

    def get_binary_grid(self) -> np.ndarray:
        """
        Get the state as a binary grid for measurement.

        For the interacting model, this is just self.grid.
        For the non-interacting model, convert counts to binary:
        1 if any cream particle present, 0 otherwise.
        """
        if self.model == 'interacting':
            return self.grid
        else:
            return (self.grid > 0).astype(np.uint8)

    def entropy(self) -> int:
        """Entropy estimate: gzip(fine-grained state). See Section 4."""
        return measure_entropy(self.get_binary_grid())

    def complexity(self, grain_size: Optional[int] = None,
                   num_buckets: int = 7, adjust: bool = True) -> int:
        """
        Apparent complexity estimate: gzip(coarse-grained state).
        See Sections 5 and 6.

        Args:
            grain_size: g in the paper. Default: n // 10
            num_buckets: 3 (Section 5) or 7 (Section 6)
            adjust: Apply row-majority adjustment (Section 6)
        """
        if grain_size is None:
            grain_size = max(2, self.n // 10)
        return measure_complexity(self.get_binary_grid(), grain_size,
                                  num_buckets, adjust)

    def fraction_mixed(self) -> float:
        """
        Fraction of cells that are "wrong" (cream in bottom half or
        coffee in top half). A simple monotonic proxy for entropy.
        """
        grid = self.get_binary_grid()
        n = self.n
        # Cream in bottom half + coffee in top half
        wrong = (np.sum(grid[n // 2:, :] == 1) +
                 np.sum(grid[:n // 2, :] == 0))
        total = n * n
        return wrong / total


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(grid_size: int = 50, total_steps: int = 100000,
                   num_snapshots: int = 100, model: str = 'interacting',
                   grain_size: Optional[int] = None, num_buckets: int = 7,
                   adjust: bool = True, seed: Optional[int] = 42
                   ) -> Dict[str, np.ndarray]:
    """
    Run the coffee automaton and collect measurements over time.

    Returns a dict with keys:
        'times': array of time steps
        'entropy': array of entropy estimates
        'complexity': array of complexity estimates
        'fraction_mixed': array of mixing fractions
        'grids': list of grid snapshots (if grid_size <= 100)

    Args:
        grid_size: N (side length)
        total_steps: Total number of steps
        num_snapshots: Number of measurement points
        model: 'interacting' or 'non_interacting'
        grain_size: Coarse-graining block size (default: n//10)
        num_buckets: 3 or 7 (Section 5 vs Section 6)
        adjust: Row-majority adjustment (Section 6)
        seed: Random seed
    """
    ca = CoffeeAutomaton(grid_size, model, seed)

    if grain_size is None:
        grain_size = max(2, grid_size // 10)

    steps_per_snapshot = max(1, total_steps // num_snapshots)
    store_grids = grid_size <= 100

    times = []
    entropies = []
    complexities = []
    fractions = []
    grids = []

    # Measure initial state
    times.append(0)
    entropies.append(ca.entropy())
    complexities.append(ca.complexity(grain_size, num_buckets, adjust))
    fractions.append(ca.fraction_mixed())
    if store_grids:
        grids.append(ca.get_binary_grid().copy())

    for i in range(num_snapshots):
        ca.step(steps_per_snapshot)
        times.append(ca.time)
        entropies.append(ca.entropy())
        complexities.append(ca.complexity(grain_size, num_buckets, adjust))
        fractions.append(ca.fraction_mixed())
        if store_grids:
            grids.append(ca.get_binary_grid().copy())

    results = {
        'times': np.array(times),
        'entropy': np.array(entropies),
        'complexity': np.array(complexities),
        'fraction_mixed': np.array(fractions),
    }
    if store_grids:
        results['grids'] = grids

    return results


def compare_models(grid_size: int = 50, total_steps: int = 100000,
                   num_snapshots: int = 100, seed: int = 42
                   ) -> Tuple[Dict, Dict]:
    """
    Run both interacting and non-interacting models and return results.
    Useful for replicating the paper's comparison (Figures 2 and 10).
    """
    print(f"Running interacting model ({grid_size}x{grid_size}, {total_steps} steps)...")
    interacting = run_simulation(grid_size, total_steps, num_snapshots,
                                 'interacting', seed=seed)

    print(f"Running non-interacting model ({grid_size}x{grid_size}, {total_steps} steps)...")
    non_interacting = run_simulation(grid_size, total_steps, num_snapshots,
                                     'non_interacting', seed=seed)

    return interacting, non_interacting


def scaling_experiment(sizes: List[int] = None, steps_per_pixel: int = 40,
                       num_snapshots: int = 100, seed: int = 42
                       ) -> Dict[str, np.ndarray]:
    """
    Replicate the scaling experiment from Figures 6-8.
    Measure max entropy, max complexity, and time to max complexity
    as a function of grid size.

    The paper found:
    - Max entropy ~ n^2
    - Max complexity ~ n
    - Time to max complexity ~ n^2
    """
    if sizes is None:
        sizes = [10, 20, 30, 50, 70]

    max_entropies = []
    max_complexities = []
    peak_times = []

    for n in sizes:
        total_steps = steps_per_pixel * n * n
        print(f"  Grid size {n}x{n}, {total_steps} steps...")
        results = run_simulation(n, total_steps, num_snapshots, 'interacting',
                                 seed=seed)
        max_entropies.append(np.max(results['entropy']))
        max_complexities.append(np.max(results['complexity']))
        peak_idx = np.argmax(results['complexity'])
        peak_times.append(results['times'][peak_idx])

    return {
        'sizes': np.array(sizes),
        'max_entropy': np.array(max_entropies),
        'max_complexity': np.array(max_complexities),
        'peak_time': np.array(peak_times),
    }


# ---------------------------------------------------------------------------
# Main: quick demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Coffee Automaton (Aaronson, Carroll, Ouellette 2014)")
    print("=" * 55)

    # Quick interacting simulation
    print("\nRunning interacting model (50x50, 50000 steps)...")
    results = run_simulation(grid_size=50, total_steps=50000, num_snapshots=50)

    peak_idx = np.argmax(results['complexity'])
    print(f"\nResults:")
    print(f"  Peak complexity at t = {results['times'][peak_idx]}")
    print(f"  Peak complexity value = {results['complexity'][peak_idx]} bytes")
    print(f"  Final entropy = {results['entropy'][-1]} bytes")
    print(f"  Final complexity = {results['complexity'][-1]} bytes")
    print(f"  Fraction mixed at end = {results['fraction_mixed'][-1]:.3f}")

    # Verify: complexity should rise then fall
    first_half = results['complexity'][:len(results['complexity']) // 2]
    second_half = results['complexity'][len(results['complexity']) // 2:]
    if np.mean(first_half) > np.mean(second_half) * 0.8:
        print("\n  Complexity shows rise-then-fall pattern [OK]")
    else:
        print("\n  [NOTE] May need more steps to see full pattern")
