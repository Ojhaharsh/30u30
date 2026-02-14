#!/usr/bin/env python3
"""
setup.py - Setup and Verification for Scaling Laws

This script:
1. Checks Python version and environment.
2. Installs required dependencies via requirements.txt.
3. Verifies the KaplanTransformer parameter math (12Ld^2).
4. Runs a quick scaling fit diagnostic using MasterFitter.

Author: 30u30 Project
License: CC BY-NC-ND 4.0
"""

import sys
import subprocess
import os
import numpy as np

def check_python_version():
    """Check if Python version is 3.8+."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"FAIL: Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("OK: Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("FAIL: Failed to install requirements")
        return False

def verify_implementation():
    """Verify that the implementation is logically sound."""
    print("\nVerifying implementation math...")
    try:
        import torch
        from implementation import KaplanTransformer, MasterFitter
        
        # 1. Verify 12Ld^2 Math
        d_model = 128
        n_layers = 2
        model = KaplanTransformer(vocab_size=100, d_model=d_model, n_heads=4, n_layers=n_layers)
        actual_n = model.count_parameters(mode="Kaplan")
        theoretical_n = 12 * n_layers * (d_model**2)
        
        if abs(actual_n - theoretical_n) / theoretical_n < 0.01:
            print(f"OK: 12Ld^2 Parameter counts verified ({actual_n:,} params)")
        else:
            print(f"FAIL: Parameter mismatch. Actual: {actual_n}, Theoretical: {theoretical_n}")
            return False

        # 2. Verify MasterFitter Logic
        ns = np.logspace(5, 8, 5)
        ls = 1.7 + (8.8e13 / ns)**0.076
        fitter = MasterFitter(ns, ls)
        fitter.fit()
        
        if abs(fitter.alpha - 0.076) < 0.01:
            print(f"OK: Power-law fitter verified (alpha={fitter.alpha:.4f})")
        else:
            print(f"FAIL: Fitter accuracy too low (alpha={fitter.alpha:.4f})")
            return False
            
        return True
    except Exception as e:
        print(f"FAIL: Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("Day 25: Scaling Laws for Neural Language Models - Setup")
    print("=" * 60)
    
    if not check_python_version():
        sys.exit(1)
    
    if not install_requirements():
        print("\nInstallation failed. Try manually: pip install -r requirements.txt")
        sys.exit(1)
    
    if not verify_implementation():
        print("\nLogic verification failed. Check implementation.py")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup complete. Day 25 is ready for exploration.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read README.md for the 'Physics of Language Models'")
    print("  2. Open notebook.ipynb for interactive analysis")
    print("  3. Run: python train_minimal.py to execute scaling simulations")

if __name__ == "__main__":
    main()
