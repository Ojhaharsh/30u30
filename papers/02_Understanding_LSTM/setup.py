#!/usr/bin/env python3
"""
Day 2 Setup Script
==================

Quick setup for Understanding LSTM Networks.

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
        print(f"❌ Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    required = ['numpy', 'matplotlib', 'jupyter']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - failed to import")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick test of the LSTM implementation."""
    print("\nRunning quick test...")
    try:
        from implementation import LSTM
        
        # Create tiny LSTM
        lstm = LSTM(input_size=5, hidden_size=8, output_size=5)
        
        # Test forward pass
        import numpy as np
        inputs = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 0]
        h = np.zeros(8)
        C = np.zeros(8)
        
        loss = lstm.forward(inputs, targets, h, C)
        
        if loss > 0:
            print(f"✅ LSTM test passed (loss: {loss:.4f})")
            return True
        else:
            print("❌ LSTM test failed")
            return False
            
    except Exception as e:
        print(f"❌ LSTM test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Day 2: Understanding LSTM Networks - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️ Installation failed. Try manually:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify imports
    if not verify_imports():
        print("\n⚠️ Some packages failed to import")
        sys.exit(1)
    
    # Run test
    if not run_quick_test():
        print("\n⚠️ Test failed. Check implementation.py")
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read README.md for overview")
    print("  2. Read paper_notes.md for ELI5 explanation")
    print("  3. Try: python train_minimal.py --data data/input.txt")
    print("  4. Open notebook.ipynb in Jupyter")
    print("  5. Try exercises in exercises/ folder")
    print("\nHappy learning! 🚀")


if __name__ == "__main__":
    main()
