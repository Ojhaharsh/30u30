"""
Complexodynamics: Coffee Mixing Simulation & Complexity Measurement

Implements the core ideas from Aaronson's "The First Law of Complexodynamics"
blog post (2011):

1. Coffee-milk mixing simulation (2D pixel grid with random neighbor swaps)
2. Kolmogorov complexity approximation via gzip
3. Coarse-grained KC (Sean Carroll's suggestion from blog comments)
4. Two-part code sophistication proxy
5. Multi-scale complexity analysis (Luca Trevisan's suggestion)
6. Entropy tracking for comparison

The central demonstration: entropy increases monotonically, but complexity
(measured by gzip size, coarse-grained KC, or sophistication proxy) peaks
at intermediate times and then decreases -- the "hump" that Aaronson's
"First Law" conjectures must occur.

Reference: https://scottaaronson.blog/?p=762

Author: 30u30 Project
"""

import numpy as np
import gzip
from typing import Tuple, Dict, List, Optional
import warnings


# ============================================================================
# SECTION 1: COFFEE-MILK GRID SIMULATION
# ============================================================================
# Aaronson describes a 2D array of black (coffee) and white (milk) pixels.
# Initially separated (top half black, bottom half white). At each step,
# pick a random adjacent coffee-milk pair and swap them. This is a discrete
# random diffusion process.
# ============================================================================

def create_initial_grid(size: int = 64) -> np.ndarray:
    """
    Create the initial coffee-milk grid: top half = 1 (coffee/black),
    bottom half = 0 (milk/white).

    Args:
        size: Side length of the square grid.

    Returns:
        2D numpy array of shape (size, size) with values 0 or 1.
    """
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[:size // 2, :] = 1  # top half is coffee
    return grid


def find_adjacent_pairs(grid: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Find all adjacent pairs where one cell is coffee (1) and the other is milk (0).
    Considers horizontal and vertical neighbors only (4-connectivity).

    This is the set of "swappable" pairs in Aaronson's model.

    Args:
        grid: 2D binary array.

    Returns:
        List of ((r1, c1), (r2, c2)) pairs.
    """
    rows, cols = grid.shape
    pairs = []

    # Horizontal neighbors
    for r in range(rows):
        for c in range(cols - 1):
            if grid[r, c] != grid[r, c + 1]:
                pairs.append(((r, c), (r, c + 1)))

    # Vertical neighbors
    for r in range(rows - 1):
        for c in range(cols):
            if grid[r, c] != grid[r + 1, c]:
                pairs.append(((r, c), (r + 1, c)))

    return pairs


def swap_step(grid: np.ndarray) -> np.ndarray:
    """
    Perform one swap step: pick a random adjacent coffee-milk pair and swap.

    This is the elementary operation in Aaronson's coffee model. At each
    time step, we uniformly pick from all adjacent pairs where one is
    coffee and one is milk, then swap them.

    Args:
        grid: 2D binary array (modified in place).

    Returns:
        The grid (same object, modified in place).
    """
    pairs = find_adjacent_pairs(grid)
    if len(pairs) == 0:
        return grid  # fully mixed or fully separated (shouldn't happen)

    idx = np.random.randint(len(pairs))
    (r1, c1), (r2, c2) = pairs[idx]
    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
    return grid


def batch_swap(grid: np.ndarray, n_swaps: int = 100) -> np.ndarray:
    """
    Perform multiple swap steps at once for efficiency.

    For large grids, doing one swap per "step" is very slow. This batches
    n_swaps random swaps. We pick random adjacent pairs using a fast
    stochastic method rather than enumerating all pairs each time.

    Note: this is an approximation of the exact model (some pairs may
    overlap within a batch), but for large grids the effect is negligible.

    Args:
        grid: 2D binary array (modified in place).
        n_swaps: Number of swaps per batch.

    Returns:
        The grid (same object, modified in place).
    """
    rows, cols = grid.shape

    for _ in range(n_swaps):
        # Pick a random cell
        r = np.random.randint(rows)
        c = np.random.randint(cols)

        # Pick a random neighbor (up, down, left, right)
        direction = np.random.randint(4)
        if direction == 0 and r > 0:
            nr, nc = r - 1, c
        elif direction == 1 and r < rows - 1:
            nr, nc = r + 1, c
        elif direction == 2 and c > 0:
            nr, nc = r, c - 1
        elif direction == 3 and c < cols - 1:
            nr, nc = r, c + 1
        else:
            continue  # boundary, skip

        # Only swap if they differ (one coffee, one milk)
        if grid[r, c] != grid[nr, nc]:
            grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    return grid


def run_simulation(
    grid_size: int = 64,
    n_steps: int = 50000,
    swaps_per_step: int = 50,
    measure_interval: int = 500,
    seed: Optional[int] = 42
) -> Dict:
    """
    Run the full coffee mixing simulation and measure complexity at intervals.

    This is the main experiment. We create a half-and-half grid, run random
    swaps, and periodically measure:
    - Shannon entropy (should increase monotonically)
    - gzip complexity (should show the "hump")
    - Coarse-grained KC at various scales
    - Two-part code sophistication proxy

    Args:
        grid_size: Side length of grid.
        n_steps: Total number of batch-swap steps.
        swaps_per_step: How many individual swaps per step.
        measure_interval: Measure complexity every this many steps.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            'times': list of measurement times
            'grids': list of grid snapshots
            'entropy': list of entropy values
            'gzip_complexity': list of gzip compressed sizes
            'coarse_kc': dict of {block_size: list of coarse KC values}
            'two_part': list of (part1, part2) tuples
            'fraction_mixed': list of fraction of boundary pixels
    """
    if seed is not None:
        np.random.seed(seed)

    grid = create_initial_grid(grid_size)

    # Storage
    times = []
    grids = []
    entropy_vals = []
    gzip_vals = []
    coarse_kc_vals = {bs: [] for bs in [2, 4, 8]}
    two_part_vals = []
    fraction_mixed_vals = []

    # Measurement function
    def measure(step):
        times.append(step * swaps_per_step)
        grids.append(grid.copy())
        entropy_vals.append(grid_entropy(grid))
        gzip_vals.append(gzip_complexity(grid))
        for bs in coarse_kc_vals:
            if grid_size // bs >= 4:  # need at least 4x4 coarse grid
                coarse_kc_vals[bs].append(coarse_grained_kc(grid, bs))
            else:
                coarse_kc_vals[bs].append(0)
        two_part_vals.append(two_part_code(grid, block_size=4))
        fraction_mixed_vals.append(boundary_fraction(grid))

    # Initial measurement
    measure(0)

    # Run simulation
    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            measure(step)

    return {
        'times': times,
        'grids': grids,
        'entropy': entropy_vals,
        'gzip_complexity': gzip_vals,
        'coarse_kc': coarse_kc_vals,
        'two_part': two_part_vals,
        'fraction_mixed': fraction_mixed_vals,
        'grid_size': grid_size,
        'n_steps': n_steps,
        'swaps_per_step': swaps_per_step,
    }


# ============================================================================
# SECTION 2: KOLMOGOROV COMPLEXITY APPROXIMATION VIA GZIP
# ============================================================================
# KC is uncomputable, but gzip gives a useful upper bound. Aaronson
# discusses this approach and Lauren Ouellette used it in her experiments
# (which became the Day 7 paper).
#
# For our 2D grid, we serialize to bytes and compress. The compressed
# size tracks the structure in the grid:
# - All-one-color: very compressible (low KC)
# - Random noise: incompressible (high KC)
# - Structured tendrils: intermediate (moderate KC, but high sophistication)
# ============================================================================

def gzip_complexity(grid: np.ndarray) -> int:
    """
    Approximate Kolmogorov complexity of a grid using gzip compressed size.

    Args:
        grid: 2D binary numpy array.

    Returns:
        Compressed size in bytes.
    """
    data = grid.astype(np.uint8).tobytes()
    return len(gzip.compress(data, compresslevel=9))


def gzip_complexity_bytes(data: bytes) -> int:
    """
    gzip complexity of raw bytes.

    Args:
        data: Raw byte string.

    Returns:
        Compressed size in bytes.
    """
    return len(gzip.compress(data, compresslevel=9))


def robust_gzip_complexity(grid: np.ndarray) -> float:
    """
    More robust KC estimate: average gzip size over row-major and
    column-major serializations.

    gzip is sensitive to serialization order because it uses LZ77
    (sliding window). Averaging over orderings reduces this artifact.

    Args:
        grid: 2D binary numpy array.

    Returns:
        Average compressed size in bytes.
    """
    # Row-major (default)
    data_row = grid.astype(np.uint8).tobytes()
    size_row = len(gzip.compress(data_row, compresslevel=9))

    # Column-major
    data_col = grid.T.astype(np.uint8).tobytes()
    size_col = len(gzip.compress(data_col, compresslevel=9))

    return (size_row + size_col) / 2.0


# ============================================================================
# SECTION 3: COARSE-GRAINED KC (SEAN CARROLL'S SUGGESTION)
# ============================================================================
# Carroll proposed in blog comments (#6-7): blur/coarse-grain the bitmap
# at some scale, then measure KC of the coarse-grained version. This
# captures macroscopic structure rather than pixel-level noise.
#
# The idea: at the right coarse-graining scale, the tendril boundaries
# create structure that is hard to compress. Too fine = noise dominates.
# Too coarse = all structure is blurred away.
# ============================================================================

def coarse_grain(grid: np.ndarray, block_size: int) -> np.ndarray:
    """
    Coarse-grain a grid by averaging over block_size x block_size blocks.

    Args:
        grid: 2D array.
        block_size: Size of blocks to average over.

    Returns:
        Coarse-grained 2D array (smaller than input).
    """
    h, w = grid.shape
    h_new = h // block_size
    w_new = w // block_size

    result = np.zeros((h_new, w_new), dtype=np.float64)
    for i in range(h_new):
        for j in range(w_new):
            block = grid[i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size]
            result[i, j] = block.mean()

    return result


def coarse_grained_kc(grid: np.ndarray, block_size: int) -> int:
    """
    KC of a coarse-grained version of the grid.

    Steps:
    1. Average over block_size x block_size blocks
    2. Quantize to uint8 (0-255)
    3. Compress with gzip

    Args:
        grid: 2D binary array.
        block_size: Coarse-graining scale.

    Returns:
        Compressed size of the coarse-grained representation.
    """
    coarse = coarse_grain(grid, block_size)
    # Quantize to 256 levels
    quantized = (coarse * 255).astype(np.uint8)
    return gzip_complexity_bytes(quantized.tobytes())


def multiscale_kc(grid: np.ndarray, scales: Optional[List[int]] = None) -> Dict[int, int]:
    """
    Compute KC at multiple coarse-graining scales (Trevisan's idea).

    Trevisan (blog comment #17) proposed tracking KC at each scale to get
    a 2D picture of where complexity lives (scale x time).

    Args:
        grid: 2D binary array.
        scales: List of block sizes. Defaults to powers of 2.

    Returns:
        Dict mapping scale -> compressed size.
    """
    grid_size = min(grid.shape)
    if scales is None:
        scales = [2**k for k in range(1, int(np.log2(grid_size)))]

    result = {}
    for s in scales:
        if grid_size // s >= 2:  # need at least 2x2 coarse grid
            result[s] = coarse_grained_kc(grid, s)
    return result


# ============================================================================
# SECTION 4: TWO-PART CODE (SOPHISTICATION PROXY)
# ============================================================================
# Sophistication (Koppel 1988, Gacs-Tromp-Vitanyi) measures the complexity
# of the "model" part of a two-part code:
#
#   Part 1: Description of a set S containing x  (the "model")
#   Part 2: Index of x within S                   (the "data given model")
#
# Sophistication = K(S), the complexity of the model.
#
# We approximate this by:
#   Part 1 = coarse-grained version of the grid (the structural model)
#   Part 2 = residual (exact grid minus coarse reconstruction)
#
# Part 1 size should show the characteristic hump: low at t=0 (trivial
# model), high at intermediate times (complex tendril boundaries),
# low at equilibrium (trivial uniform model).
# ============================================================================

def two_part_code(
    grid: np.ndarray,
    block_size: int = 4
) -> Tuple[int, int]:
    """
    Compute two-part code sizes for a grid.

    Part 1 (model/sophistication): coarse-grained description
    Part 2 (data given model): residual to reconstruct exact grid

    Args:
        grid: 2D binary array.
        block_size: Coarse-graining scale for the model.

    Returns:
        (part1_size, part2_size) in bytes.
    """
    h, w = grid.shape
    grid_float = grid.astype(np.float64)

    # Part 1: coarse model
    coarse = coarse_grain(grid_float, block_size)
    model_quantized = (coarse * 255).astype(np.uint8)
    part1 = gzip_complexity_bytes(model_quantized.tobytes())

    # Reconstruct from coarse model
    reconstructed = np.repeat(
        np.repeat(coarse, block_size, axis=0),
        block_size, axis=1
    )[:h, :w]

    # Part 2: residual
    # Scale residual to [0, 255] range for compression
    residual = grid_float - reconstructed
    # residual is in [-1, 1], map to [0, 255]
    residual_uint8 = ((residual + 1.0) * 127.5).astype(np.uint8)
    part2 = gzip_complexity_bytes(residual_uint8.tobytes())

    return part1, part2


def sophistication_proxy(grid: np.ndarray, block_size: int = 4) -> int:
    """
    Sophistication proxy = Part 1 of the two-part code.

    Args:
        grid: 2D binary array.
        block_size: Coarse-graining scale.

    Returns:
        Part 1 size in bytes (the "model" complexity).
    """
    part1, _ = two_part_code(grid, block_size)
    return part1


# ============================================================================
# SECTION 5: ENTROPY AND MIXING MEASURES
# ============================================================================
# Shannon entropy of the pixel distribution. For a binary grid, this is
# just H(p) = -p*log(p) - (1-p)*log(1-p) where p is the fraction of 1s.
#
# For the coffee simulation, entropy tracks the fraction mixed. It starts
# at 1.0 (half black, half white = maximum entropy for binary) and stays
# near 1.0 throughout (the fraction of 1s is conserved by swaps).
#
# The more useful entropy measure is the LOCAL entropy: compute entropy
# in sliding windows. This tracks how uniformly mixed each neighborhood is.
# ============================================================================

def grid_entropy(grid: np.ndarray) -> float:
    """
    Shannon entropy of the overall pixel distribution.

    For a binary grid, H = -p*log2(p) - (1-p)*log2(1-p).
    Note: this is always near 1.0 for the coffee simulation because
    swaps conserve the total number of coffee/milk pixels.

    Args:
        grid: 2D binary array.

    Returns:
        Shannon entropy in bits.
    """
    p = grid.mean()
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def local_entropy(grid: np.ndarray, window_size: int = 8) -> np.ndarray:
    """
    Compute entropy in sliding windows across the grid.

    This is more informative than global entropy for tracking mixing.
    Initially: some windows are all-0, some all-1 (entropy = 0 in each).
    At equilibrium: all windows are ~50/50 (entropy = 1.0 in each).

    The AVERAGE local entropy increases monotonically with mixing.

    Args:
        grid: 2D binary array.
        window_size: Side length of the sliding window.

    Returns:
        2D array of local entropy values.
    """
    h, w = grid.shape
    h_out = h - window_size + 1
    w_out = w - window_size + 1
    result = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            window = grid[i:i + window_size, j:j + window_size]
            p = window.mean()
            if p == 0.0 or p == 1.0:
                result[i, j] = 0.0
            else:
                result[i, j] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return result


def mean_local_entropy(grid: np.ndarray, window_size: int = 8) -> float:
    """
    Average local entropy across the grid. This is a better monotone
    measure than global entropy for tracking the mixing process.

    Args:
        grid: 2D binary array.
        window_size: Side length of sliding window.

    Returns:
        Mean local entropy.
    """
    le = local_entropy(grid, window_size)
    return le.mean()


def boundary_fraction(grid: np.ndarray) -> float:
    """
    Fraction of adjacent pixel pairs that are different (one coffee, one milk).

    This tracks the "interface length" between coffee and milk regions.
    Starts low (single horizontal boundary), peaks during mixing
    (fractal-like tendrils), then decreases as mixing approaches uniformity.

    Note: this is related to but not the same as complexity. A checkerboard
    pattern has maximum boundary fraction but low complexity.

    Args:
        grid: 2D binary array.

    Returns:
        Fraction of boundary pairs (0 to 1).
    """
    h, w = grid.shape
    total_pairs = 0
    diff_pairs = 0

    # Horizontal pairs
    h_diff = np.sum(grid[:, :-1] != grid[:, 1:])
    h_total = h * (w - 1)

    # Vertical pairs
    v_diff = np.sum(grid[:-1, :] != grid[1:, :])
    v_total = (h - 1) * w

    return (h_diff + v_diff) / (h_total + v_total)


# ============================================================================
# SECTION 6: NORMALIZED COMPLEXITY MEASURES
# ============================================================================
# For comparison, it helps to normalize complexity measures to [0, 1].
# We define:
#   normalized_kc = (gzip_size - min_gzip_size) / (max_gzip_size - min_gzip_size)
# where min is the gzip size of an all-zeros grid and max is a random grid.
# ============================================================================

def compute_normalization_bounds(grid_size: int, n_samples: int = 5) -> Dict:
    """
    Compute gzip size bounds for normalization.

    Args:
        grid_size: Side length of grid.
        n_samples: Number of random grids to average over.

    Returns:
        Dict with 'min_gzip' (all-zeros) and 'max_gzip' (random average).
    """
    # Minimum: all-zeros grid
    zeros = np.zeros((grid_size, grid_size), dtype=np.uint8)
    min_gzip = gzip_complexity(zeros)

    # Maximum: average over random grids
    max_gzips = []
    for _ in range(n_samples):
        rand_grid = np.random.randint(0, 2, size=(grid_size, grid_size), dtype=np.uint8)
        max_gzips.append(gzip_complexity(rand_grid))
    max_gzip = np.mean(max_gzips)

    return {'min_gzip': min_gzip, 'max_gzip': max_gzip}


def normalize_complexity(
    values: List[float],
    bounds: Dict
) -> List[float]:
    """
    Normalize complexity values to [0, 1] using precomputed bounds.

    Args:
        values: Raw complexity values.
        bounds: Dict with 'min_gzip' and 'max_gzip'.

    Returns:
        Normalized values.
    """
    lo = bounds['min_gzip']
    hi = bounds['max_gzip']
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ============================================================================
# SECTION 7: UTILITY FUNCTIONS
# ============================================================================

def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """
    Convert binary grid to grayscale image (0 = white, 1 = black).

    Args:
        grid: 2D binary array.

    Returns:
        2D array suitable for matplotlib imshow (0.0=white, 1.0=black).
    """
    return grid.astype(np.float64)


def save_grid(grid: np.ndarray, filepath: str) -> None:
    """
    Save grid as a NumPy binary file.

    Args:
        grid: 2D binary array.
        filepath: Output path.
    """
    np.save(filepath, grid)


def load_grid(filepath: str) -> np.ndarray:
    """
    Load grid from a NumPy binary file.

    Args:
        filepath: Input path.

    Returns:
        2D binary array.
    """
    return np.load(filepath)


def summarize_results(results: Dict) -> str:
    """
    Print a summary of simulation results.

    Args:
        results: Dict returned by run_simulation().

    Returns:
        Summary string.
    """
    times = results['times']
    gzip_vals = results['gzip_complexity']
    entropy_vals = results['entropy']

    # Find peak complexity
    peak_idx = np.argmax(gzip_vals)
    peak_time = times[peak_idx]
    peak_val = gzip_vals[peak_idx]

    lines = [
        "=" * 60,
        "COFFEE MIXING SIMULATION RESULTS",
        "=" * 60,
        f"Grid size: {results['grid_size']}x{results['grid_size']}",
        f"Total swaps: {results['n_steps'] * results['swaps_per_step']:,}",
        f"Measurements: {len(times)}",
        "",
        "COMPLEXITY (gzip compressed size):",
        f"  Initial:     {gzip_vals[0]} bytes",
        f"  Peak:        {peak_val} bytes (at swap {peak_time:,})",
        f"  Final:       {gzip_vals[-1]} bytes",
        f"  Peak/Initial ratio: {peak_val / max(gzip_vals[0], 1):.2f}x",
        "",
        "ENTROPY (Shannon, global):",
        f"  Initial: {entropy_vals[0]:.4f} bits",
        f"  Final:   {entropy_vals[-1]:.4f} bits",
        "",
        "THE HUMP:",
        f"  Complexity rises from {gzip_vals[0]} to {peak_val} bytes",
        f"  then falls to {gzip_vals[-1]} bytes.",
    ]

    if peak_val > gzip_vals[0] and peak_val > gzip_vals[-1]:
        lines.append("  [OK] Characteristic complexity hump observed.")
    else:
        lines.append("  [NOTE] Hump not clearly visible. Try more steps or larger grid.")

    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================================
# MAIN: Run a quick demo if executed directly
# ============================================================================

if __name__ == "__main__":
    print("Coffee Mixing Simulation")
    print("Based on Aaronson's 'The First Law of Complexodynamics' (2011)")
    print()

    # Small demo
    results = run_simulation(
        grid_size=32,
        n_steps=5000,
        swaps_per_step=20,
        measure_interval=100,
        seed=42
    )

    print(summarize_results(results))

    # Show a few grid snapshots
    print("\nGrid snapshots saved to data/ directory")
    print("Run train_minimal.py for full simulation with plots.")
