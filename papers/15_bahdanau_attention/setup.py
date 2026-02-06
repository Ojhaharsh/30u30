#!/usr/bin/env python3
"""
Day 15 Setup Script
===================

Quick setup for Bahdanau Attention (Neural Machine Translation).

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
        print(f"Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install requirements")
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    required = ['torch', 'numpy', 'matplotlib', 'jupyter']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            print(f"{package} - failed to import")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick test of the Attention implementation."""
    print("\nRunning quick test...")
    try:
        import torch
        from implementation import BahdanauAttention
        
        # Create tiny Attention module
        attention = BahdanauAttention(encoder_dim=16, decoder_dim=16)
        
        # Test forward pass
        query = torch.randn(2, 16)
        keys = torch.randn(2, 5, 16)
        
        context, weights = attention(query, keys)
        
        if context.shape == (2, 16) and weights.shape == (2, 5):
            print("Attention test passed")
            return True
        else:
            print(f"Attention test failed: shapes {context.shape}, {weights.shape}")
            return False
            
    except Exception as e:
        print(f"Attention test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Day 15: Bahdanau Attention - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\nInstallation failed. Try manually:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify imports
    if not verify_imports():
        # jupyter is optional for terminal users
        print("\nNote: Some packages (like jupyter) failed to import. Notebook might not work.")
    
    # Run test
    if not run_quick_test():
        print("\nTest failed. Check implementation.py")
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("Setup complete.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read README.md for overview")
    print("  2. Read paper_notes.md for ELI5 explanation")
    print("  3. Try: python train_minimal.py --data reversal")
    print("  4. Open notebook.ipynb in Jupyter")
    print("  5. Try exercises in exercises/ folder")
    print("\nSetup finished. See README.md for next steps.")


if __name__ == "__main__":
    main()
