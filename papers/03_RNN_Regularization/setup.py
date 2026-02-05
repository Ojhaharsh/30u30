#!/usr/bin/env python3
"""
Day 3 Setup Script
==================

Quick setup for RNN Regularization & Generalization.

This script:
1. Checks Python version
2. Installs required packages
3. Verifies installation
4. Runs a quick test

Usage:
    python setup.py
"""

import sys
import subprocess
import os


def check_python_version():
    """Check if Python version is 3.7+."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"[FAIL] Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[OK] Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("[FAIL] Failed to install requirements")
        print("   Try: pip install -r requirements.txt")
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    required = ['numpy', 'matplotlib', 'jupyter']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} - failed to import")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick test of the regularization implementation."""
    print("\nRunning quick test...")
    try:
        from implementation import dropout_forward, layer_norm_forward, compute_l2_penalty
        import numpy as np
        
        # Test dropout
        x = np.random.randn(10, 5)
        out, mask = dropout_forward(x, keep_prob=0.8, training=True)
        assert out.shape == x.shape, "Dropout shape mismatch"
        print("[OK] Dropout forward pass")
        
        # Test layer norm
        gamma = np.ones(5)
        beta = np.zeros(5)
        norm_out, cache = layer_norm_forward(x, gamma, beta)
        assert norm_out.shape == x.shape, "Layer norm shape mismatch"
        print("[OK] Layer normalization")
        
        # Test weight decay
        weights = [np.random.randn(10, 10)]
        penalty = compute_l2_penalty(weights, weight_decay=0.01)
        assert penalty > 0, "Weight decay should be positive"
        print("[OK] Weight decay penalty")
        
        # Test early stopping
        from implementation import EarlyStoppingMonitor
        monitor = EarlyStoppingMonitor(patience=3, verbose=False)
        assert monitor.check(1.0, 0) == True, "First check should continue"
        print("[OK] Early stopping monitor")
        
        print("\n[OK] All quick tests passed!")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except AssertionError as e:
        print(f"[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def check_data():
    """Check if sample data exists."""
    print("\nChecking sample data...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_text.txt')
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            content = f.read()
        print(f"[OK] Sample data found ({len(content)} characters)")
        return True
    else:
        print("[NOTE] Sample data not found (optional)")
        return True


def main():
    """Run all setup checks."""
    print("\n" + "=" * 60)
    print("Day 3: RNN Regularization & Generalization Setup")
    print("=" * 60 + "\n")
    
    # Run all checks
    results = []
    
    results.append(("Python version", check_python_version()))
    results.append(("Requirements", install_requirements()))
    results.append(("Imports", verify_imports()))
    results.append(("Quick tests", run_quick_test()))
    results.append(("Sample data", check_data()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "-" * 60)
    
    if all_passed:
        print("[OK] All checks passed! Ready to learn regularization.")
        print("\nNext steps:")
        print("  1. Read README.md for overview")
        print("  2. Open notebook.ipynb for interactive learning")
        print("  3. Try: python train_minimal.py --epochs 10")
        print("  4. Work through exercises/ in order")
    else:
        print("[NOTE] Some checks failed. Please fix the issues above.")
        return 1
    
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
