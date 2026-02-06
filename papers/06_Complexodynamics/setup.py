"""
setup.py - Verify environment for Day 6: The First Law of Complexodynamics

Run this to check that all dependencies are available:
    python setup.py

Reference: Scott Aaronson (2011) - https://scottaaronson.blog/?p=762
"""

import sys


def check_dependencies():
    """Check that required packages are installed."""
    required = {
        'numpy': 'numpy',
        'gzip': 'gzip',      # stdlib
        'matplotlib': 'matplotlib',
    }

    missing = []
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} -- install with: pip install {name}")
            missing.append(name)

    return missing


def check_implementation():
    """Verify core implementation imports work."""
    try:
        from implementation import (
            create_initial_grid,
            gzip_complexity,
            coarse_grained_kc,
            two_part_code,
            run_simulation,
        )
        print("  [OK] implementation.py imports")
    except ImportError as e:
        print(f"  [FAIL] implementation.py: {e}")
        return False

    # Quick smoke test
    try:
        import numpy as np
        grid = create_initial_grid(16)
        assert grid.shape == (16, 16), f"Expected (16,16), got {grid.shape}"
        assert grid[:8, :].sum() == 16 * 8, "Top half should be all 1s"
        assert grid[8:, :].sum() == 0, "Bottom half should be all 0s"

        kc = gzip_complexity(grid)
        assert isinstance(kc, int) and kc > 0, f"gzip_complexity returned {kc}"

        print("  [OK] Smoke test passed")
    except Exception as e:
        print(f"  [FAIL] Smoke test: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Day 6: The First Law of Complexodynamics")
    print("Setup Verification")
    print("=" * 50)
    print()

    print("Checking dependencies...")
    missing = check_dependencies()
    print()

    if missing:
        stdlib = {'gzip'}
        real_missing = [m for m in missing if m not in stdlib]
        if real_missing:
            print(f"Missing packages: {', '.join(real_missing)}")
            print(f"Install with: pip install {' '.join(real_missing)}")
            sys.exit(1)

    print("Checking implementation...")
    if check_implementation():
        print()
        print("All checks passed. Ready to go.")
        print()
        print("Quick start:")
        print("  python train_minimal.py --grid-size 64 --steps 50000")
        print()
        print("Or run the full visualization suite:")
        print("  python visualization.py")
    else:
        print()
        print("Some checks failed. See errors above.")
        sys.exit(1)
