"""
setup.py - Day 21: Neural Message Passing for Quantum Chemistry

Environment setup: check dependencies, set seeds, detect device.
Run this first to verify your environment is ready.

Usage:
    python setup.py
"""

import sys
import subprocess


def check_dependency(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  [OK] {package_name} ({version})")
        return True
    except ImportError:
        print(f"  [MISSING] {package_name} — install with: pip install {package_name}")
        return False


def check_torch_details():
    """Print PyTorch-specific details (CUDA, device)."""
    try:
        import torch
        print(f"\n  PyTorch details:")
        print(f"    Version: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    Using device: {device}")
    except ImportError:
        pass


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        print(f"  NumPy seed set to {seed}")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  PyTorch seed set to {seed}")
    except ImportError:
        pass


def main():
    print("Day 21 Setup: Neural Message Passing for Quantum Chemistry")
    print("Gilmer et al. 2017 — arXiv:1704.01212")
    print("=" * 55)

    print("\nChecking dependencies:")
    all_ok = True
    all_ok &= check_dependency('torch')
    all_ok &= check_dependency('numpy')
    all_ok &= check_dependency('matplotlib')
    all_ok &= check_dependency('networkx')

    if not all_ok:
        print("\nSome dependencies are missing. Install them with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    check_torch_details()

    print("\nSetting seeds:")
    set_seeds(42)

    # Quick import test
    print("\nTesting implementation import:")
    try:
        from implementation import MPNN, generate_dataset, ATOM_TYPES, BOND_TYPES
        print(f"  [OK] MPNN class imported")
        print(f"  [OK] Atom types: {ATOM_TYPES}")
        print(f"  [OK] Bond types: {BOND_TYPES}")

        # Quick smoke test
        dataset = generate_dataset(n_molecules=5, n_targets=1, seed=42)
        print(f"  [OK] Generated {len(dataset)} test molecules")

        model = MPNN(
            node_dim=len(ATOM_TYPES),
            edge_dim=len(BOND_TYPES),
            hidden_dim=32,
            output_dim=1,
            n_messages=2
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] MPNN model created ({total_params:,} parameters)")

    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        sys.exit(1)

    print("\nSetup complete. You are ready to go.")
    print("  Next: python train_minimal.py --epochs 5")


if __name__ == '__main__':
    main()
