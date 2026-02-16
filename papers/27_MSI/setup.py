#!/usr/bin/env python3
"""
setup.py - Verify environment for Day 27: Machine Super Intelligence

Run this to check that all dependencies are available:
    python setup.py

Reference: Shane Legg (2008) - http://www.vetta.org/documents/Machine_Super_Intelligence.pdf
"""

import sys
import subprocess
import os


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
    if not os.path.exists("requirements.txt"):
        print("NOTE: requirements.txt not found, skipping installation.")
        return True
        
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
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    required = ['numpy', 'matplotlib']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"OK: {package}")
        except ImportError:
            print(f"FAIL: {package} - failed to import")
            all_ok = False
    
    return all_ok


def run_functional_test():
    """Run a quick test of the MSI implementation."""
    print("\nRunning functional verification test...")
    try:
        from implementation import GridWorld, RandomAgent, UniversalIntelligenceMeasure
        
        # Create a small environment
        env = GridWorld(size=3)
        agent = RandomAgent()
        measure = UniversalIntelligenceMeasure([env])
        
        # Test evaluation
        res = measure.evaluate(agent, episodes=1)
        
        if res['upsilon_raw'] >= 0:
            print(f"OK: Functional test passed (Raw Upsilon: {res['upsilon_raw']:.4f})")
            return True
        else:
            print("FAIL: Functional test failed")
            return False
            
    except Exception as e:
        print(f"FAIL: Functional test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Day 27: Machine Super Intelligence - Setup")
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
        print("\nSome packages failed to import")
        sys.exit(1)
    
    # Run test
    if not run_functional_test():
        print("\nTest failed. Check implementation.py")
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("Setup complete. You are ready to explore Day 27.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read README.md for overview")
    print("  2. Read paper_notes.md for theoretical depth")
    print("  3. Try: python train_minimal.py")
    print("  4. Try: python visualization.py")
    print("  5. Explore the exercises in exercises/ folder")


if __name__ == "__main__":
    main()
