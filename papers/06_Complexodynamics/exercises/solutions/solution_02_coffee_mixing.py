"""
Solution 2: Coffee Mixing Simulation
"""

import numpy as np


def create_grid(size: int = 64) -> np.ndarray:
    """Create initial coffee-milk grid."""
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[:size // 2, :] = 1  # top half is coffee
    return grid


def swap_step(grid: np.ndarray) -> np.ndarray:
    """Perform one random swap of adjacent coffee-milk pair."""
    rows, cols = grid.shape
    pairs = []

    # Find all swappable horizontal pairs
    for r in range(rows):
        for c in range(cols - 1):
            if grid[r, c] != grid[r, c + 1]:
                pairs.append(((r, c), (r, c + 1)))

    # Find all swappable vertical pairs
    for r in range(rows - 1):
        for c in range(cols):
            if grid[r, c] != grid[r + 1, c]:
                pairs.append(((r, c), (r + 1, c)))

    if not pairs:
        return grid

    idx = np.random.randint(len(pairs))
    (r1, c1), (r2, c2) = pairs[idx]
    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
    return grid


def batch_swap(grid: np.ndarray, n_swaps: int = 100) -> np.ndarray:
    """Perform n_swaps random swaps efficiently."""
    rows, cols = grid.shape

    for _ in range(n_swaps):
        r = np.random.randint(rows)
        c = np.random.randint(cols)
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
            continue

        if grid[r, c] != grid[nr, nc]:
            grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    return grid


def run_and_snapshot(grid_size=32, n_steps=5000, swaps_per_step=20,
                     snapshot_interval=500):
    """Run simulation and collect grid snapshots."""
    np.random.seed(42)
    grid = create_grid(grid_size)
    snapshots = [(0, grid.copy())]

    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % snapshot_interval == 0:
            snapshots.append((step, grid.copy()))

    return snapshots


if __name__ == "__main__":
    print("Solution 2: Coffee Mixing Simulation")
    print("=" * 40)
    print()

    snapshots = run_and_snapshot()
    print(f"Collected {len(snapshots)} snapshots")

    # Verify initial state
    _, g0 = snapshots[0]
    assert g0[:16, :].sum() == 32 * 16, "Top half should be all coffee"
    assert g0[16:, :].sum() == 0, "Bottom half should be all milk"
    print("[OK] Initial state verified")

    # Verify mixing occurred
    _, g_final = snapshots[-1]
    top_fraction = g_final[:16, :].mean()
    assert 0.3 < top_fraction < 0.7, f"Expected ~0.5, got {top_fraction}"
    print(f"[OK] Mixing verified: top half fraction = {top_fraction:.3f}")

    try:
        import matplotlib.pyplot as plt
        n = min(6, len(snapshots))
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)
        for ax, idx in zip(axes, indices):
            step, grid = snapshots[idx]
            ax.imshow(grid, cmap='gray', interpolation='nearest')
            ax.set_title(f"step={step}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("coffee_mixing_snapshots.png", dpi=100)
        print("Saved: coffee_mixing_snapshots.png")
    except ImportError:
        print("(matplotlib not available)")
