"""
setup.py - Environment verification for Day 29 (PPO)

Checks that all required dependencies are installed and working.
"""

import sys


def check_dependency(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {name} {version}")
        return True
    except ImportError:
        print(f"  [FAIL] {name} not found. Install with: pip install {name}")
        return False


def main():
    print("Day 29: PPO Environment Check")
    print("=" * 40)

    all_ok = True
    all_ok &= check_dependency("numpy")
    all_ok &= check_dependency("torch")
    all_ok &= check_dependency("matplotlib")

    # Check gymnasium (preferred) or gym (fallback)
    try:
        import gymnasium
        print(f"  [OK] gymnasium {gymnasium.__version__}")
    except ImportError:
        try:
            import gym
            print(f"  [OK] gym {gym.__version__} (gymnasium preferred)")
        except ImportError:
            print("  [FAIL] gymnasium not found. Install with: pip install gymnasium")
            all_ok = False

    print()
    if all_ok:
        print("[OK] All dependencies satisfied.")
        print("Run: python implementation.py --demo")
    else:
        print("[FAIL] Some dependencies missing. Install with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
