#!/usr/bin/env python3
""" Day 26: Kolmogorov Complexity | Setup Script | Part of 30u30 """
"""
Day 26 Setup Script
==================

This script:
1. Checks Python version
2. Installs required packages
3. Verifies installation
4. Runs a quick functional test of estimators

Usage:
    python setup.py
"""

import sys
import subprocess
import os

def check_python_version():
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"FAIL: Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
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

def verify_imports():
    print("\nVerifying imports...")
    required = ['numpy', 'matplotlib', 'jupyter']
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"OK: {package}")
        except ImportError:
            print(f"FAIL: {package} - failed to import")
            all_ok = False
    return all_ok

def run_quick_test():
    print("\nRunning quick functional test...")
    try:
        from implementation import HuffmanCoder, ArithmeticCoder
        
        test_str = "KOLMOGOROV"
        
        # Test Huffman
        h_coder = HuffmanCoder()
        h_bits = h_coder.get_complexity(test_str)
        
        # Test Arithmetic
        a_coder = ArithmeticCoder()
        val, length = a_coder.encode(test_str)
        
        if h_bits > 0 and 0.0 <= val <= 1.0:
            print(f"OK: Huffman bits: {h_bits}, Arithmetic val: {val:.4f}")
            return True
        else:
            print("FAIL: Functional test returned invalid values")
            return False
            
    except Exception as e:
        print(f"FAIL: Functional test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Day 26: Kolmogorov Complexity - Setup")
    print("=" * 60)
    
    if not check_python_version():
        sys.exit(1)
    
    if not install_requirements():
        print("\nInstallation failed. Try manually: pip install -r requirements.txt")
        sys.exit(1)
    
    if not verify_imports():
        print("\nSome packages failed to import")
        sys.exit(1)
    
    if not run_quick_test():
        print("\nFunctional test failed. Check implementation.py")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup complete.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read README.md for overview")
    print("  2. Explore paper_notes.md for theoretical breakdown")
    print("  3. Run: python visualization.py to see complexity plots")
    print("  4. Open notebook.ipynb in Jupyter for interactive traces")
    print("  5. Try exercises in exercises/ folder")

if __name__ == "__main__":
    main()
