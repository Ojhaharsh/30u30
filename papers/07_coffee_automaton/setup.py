"""
setup.py — Verify dependencies for Day 7: The Coffee Automaton

Required: numpy, matplotlib (that's it — no scipy needed)
The paper's model uses only binary grids and gzip compression.
"""

import subprocess
import sys


def check_package(name):
    """Check if a package is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main():
    required = ['numpy', 'matplotlib']
    builtin = ['gzip', 'bz2', 'lzma', 'zlib']  # used for compression, all in stdlib

    print("Day 7: The Coffee Automaton — Dependency Check")
    print("=" * 50)

    all_ok = True

    # Check required packages
    for pkg in required:
        if check_package(pkg):
            print(f"  [OK] {pkg}")
        else:
            print(f"  [MISSING] {pkg}")
            all_ok = False

    # Check stdlib modules (should always be available)
    for pkg in builtin:
        if check_package(pkg):
            print(f"  [OK] {pkg} (stdlib)")
        else:
            print(f"  [MISSING] {pkg} — this shouldn't happen (stdlib)")
            all_ok = False

    if all_ok:
        print("\nAll dependencies satisfied. Ready to go.")
        print("\nQuick start:")
        print("  python train_minimal.py --grid-size 50 --steps 100000")
    else:
        print("\nMissing packages. Install with:")
        print("  pip install -r requirements.txt")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
