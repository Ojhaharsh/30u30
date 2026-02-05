#!/usr/bin/env python3
"""
Day 4 Setup Script
==================

Quick setup for MDL & Bayesian Neural Networks.

This script:
1. Checks Python version
2. Installs required packages (including SciPy!)
3. Verifies installation
4. Runs a quick test of the Bayesian Layer

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
        ])
        print("[OK] Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("[FAIL] Failed to install requirements")
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    required = ['numpy', 'matplotlib', 'scipy', 'seaborn', 'jupyter']
    
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
    """Run a quick test of the MDL Network implementation."""
    print("\nRunning quick test...")
    try:
        from implementation import MDLNetwork
        import numpy as np
        
        # Create tiny Bayesian Network
        # [1 input] -> [5 hidden] -> [1 output]
        net = MDLNetwork(input_size=1, hidden_size=5, output_size=1)
        
        # Create dummy data
        x = np.array([[0.5]])
        
        # Test forward pass (Sampling weights)
        y1 = net.forward(x)
        y2 = net.forward(x) # Should be slightly different due to noise!
        
        # Test KL Divergence calculation
        kl = net.total_kl()
        
        print(f"[OK] MDL Network Initialized")
        print(f"   - Forward pass 1: {y1.item():.4f}")
        print(f"   - Forward pass 2: {y2.item():.4f} (Note: Noise is active!)")
        print(f"   - Complexity Cost (KL): {kl:.4f}")
        
        if y1.shape == (1, 1) and kl != 0:
            print("[OK] Test passed")
            return True
        else:
            print("[FAIL] Dimensions or KL calculation incorrect")
            return False
            
    except Exception as e:
        print(f"[FAIL] MDL test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Day 4: MDL & Bayesian Networks - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n[NOTE] Installation failed. Try manually:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify imports
    if not verify_imports():
        print("\n[NOTE] Some packages failed to import")
        sys.exit(1)
    
    # Run test
    if not run_quick_test():
        print("\n[NOTE] Test failed. Check implementation.py")
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("[OK] Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read README.md for the theory")
    print("  2. Read paper_notes.md for the ELI5")
    print("  3. Run the experiment: python train_minimal.py")
    print("  4. Open notebook.ipynb to see the Uncertainty Envelope!")
    print("\nReady. Run: python train_minimal.py")


if __name__ == "__main__":
    main()