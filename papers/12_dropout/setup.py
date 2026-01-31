"""
Setup script for Day 12: Dropout

Installs dependencies and verifies the environment.
"""

import subprocess
import sys


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
    ])
    print("✓ Requirements installed!")


def verify_installation():
    """Verify all dependencies are available."""
    print("\nVerifying installation...")
    
    required = ['numpy', 'matplotlib']
    optional = ['torch', 'tqdm', 'jupyter']
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (required)")
            missing.append(pkg)
    
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} (optional)")
        except ImportError:
            print(f"  ○ {pkg} (optional, not installed)")
    
    if missing:
        print(f"\n❌ Missing required packages: {missing}")
        return False
    
    print("\n✓ All required packages available!")
    return True


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\nRunning quick test...")
    
    import numpy as np
    from implementation import Dropout
    
    # Test dropout
    dropout = Dropout(p=0.5)
    x = np.ones((4, 8))
    
    dropout.training = True
    y_train = dropout.forward(x)
    
    dropout.training = False
    y_eval = dropout.forward(x)
    
    # Verify
    assert np.any(y_train == 0), "Training should have some zeros"
    assert np.array_equal(y_eval, x), "Eval should be unchanged"
    
    print("  ✓ Dropout forward pass works")
    
    # Test backward
    dropout.training = True
    y = dropout.forward(x)
    grad = dropout.backward(np.ones_like(y))
    
    assert grad.shape == x.shape, "Gradient shape mismatch"
    print("  ✓ Dropout backward pass works")
    
    print("\n✓ Quick test passed!")


def main():
    """Setup and verify Day 12 environment."""
    print("=" * 50)
    print("DAY 12: DROPOUT - SETUP")
    print("=" * 50)
    
    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"Warning: Could not install requirements: {e}")
    
    # Verify installation
    if not verify_installation():
        print("\nPlease install missing packages and try again.")
        return
    
    # Run quick test
    try:
        run_quick_test()
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Read README.md for the full tutorial")
    print("  2. Run: python implementation.py")
    print("  3. Run: python train_minimal.py --epochs 10")
    print("  4. Try the exercises in exercises/")


if __name__ == "__main__":
    main()
