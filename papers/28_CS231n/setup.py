"""
setup.py - Verify environment for Day 28: CS231n CNNs for Visual Recognition

Run this to check that all dependencies are available:
    python setup.py

Reference: CS231n Course Notes - https://cs231n.github.io/convolutional-networks/
"""

import sys


def check_dependencies():
    """Check that required packages are installed."""
    required = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
    }

    optional = {
        'torch': 'torch (optional, for exercises 4-5)',
    }

    print("=" * 60)
    print("Day 28: CS231n CNNs for Visual Recognition - Setup")
    print("=" * 60)

    # Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 7):
        print("[NOTE] Python 3.7+ recommended")
    else:
        print("[OK] Python version")

    # Required packages
    print("\nRequired packages:")
    all_ok = True
    for display_name, import_name in required.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [OK] {display_name} ({version})")
        except ImportError:
            print(f"  [FAIL] {display_name} — install with: pip install {import_name}")
            all_ok = False

    # Optional packages
    print("\nOptional packages:")
    for display_name, desc in optional.items():
        try:
            mod = __import__(display_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [OK] {desc} ({version})")
        except ImportError:
            print(f"  [--] {desc} — not installed (only needed for some exercises)")

    # Quick test
    print("\nQuick test:")
    try:
        import numpy as np
        # Test the output size formula from CS231n
        W, F, P, S = 32, 3, 1, 1
        output_size = (W - F + 2 * P) // S + 1
        assert output_size == 32, f"Expected 32, got {output_size}"

        # Test convolution shapes
        x = np.random.randn(1, 3, 8, 8)
        w = np.random.randn(4, 3, 3, 3)
        assert x.shape == (1, 3, 8, 8)
        assert w.shape == (4, 3, 3, 3)

        print("  [OK] Output size formula: (32 - 3 + 2*1)/1 + 1 = 32")
        print("  [OK] NumPy array operations working")
    except Exception as e:
        print(f"  [FAIL] Quick test failed: {e}")
        all_ok = False

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("Setup complete. Ready for Day 28.")
    else:
        print("Some dependencies are missing. Install them and re-run.")

    print("\nNext steps:")
    print("  1. Read README.md for overview")
    print("  2. Read paper_notes.md for ELI5 explanation")
    print("  3. Try: python implementation.py --demo")
    print("  4. Try: python visualization.py")
    print("  5. Try exercises in exercises/ folder")
    print("=" * 60)


if __name__ == "__main__":
    check_dependencies()
