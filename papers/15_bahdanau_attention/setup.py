"""
Setup script for Day 15: Bahdanau Attention

This script helps you set up and verify your environment for the exercises.
Run this first to make sure everything is working!
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version is 3.7+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.7+)")
        return False


def check_pytorch():
    """Check PyTorch installation"""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available (CPU only - training will be slower)")
        
        return True
    except ImportError:
        print("  ✗ PyTorch not installed")
        print("    Install with: pip install torch")
        return False


def check_numpy():
    """Check NumPy installation"""
    print("\nChecking NumPy...")
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
        return True
    except ImportError:
        print("  ✗ NumPy not installed")
        print("    Install with: pip install numpy")
        return False


def check_matplotlib():
    """Check Matplotlib installation (optional, for visualization)"""
    print("\nChecking Matplotlib (optional)...")
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
        return True
    except ImportError:
        print("  ⚠ Matplotlib not installed (needed for Exercise 5)")
        print("    Install with: pip install matplotlib")
        return False


def verify_file_structure():
    """Verify all required files exist"""
    print("\nVerifying file structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        "README.md",
        "implementation.py",
        "train.py",
        "requirements.txt",
        "CHEATSHEET.md",
        "PAPER_NOTES.md",
        "exercises/README.md",
        "exercises/exercise_1.py",
        "exercises/exercise_2.py",
        "exercises/exercise_3.py",
        "exercises/exercise_4.py",
        "exercises/exercise_5.py",
        "solutions/README.md",
        "solutions/solution_1.py",
        "solutions/solution_2.py",
        "solutions/solution_3.py",
        "solutions/solution_4.py",
        "solutions/solution_5.py",
        "data/README.md",
    ]
    
    all_present = True
    for file in required_files:
        path = base_path / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            all_present = False
    
    return all_present


def test_basic_imports():
    """Test that our implementation can be imported"""
    print("\nTesting imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        from implementation import (
            BahdanauAttention,
            Encoder,
            AttentionDecoder,
            Seq2SeqWithAttention
        )
        print("  ✓ implementation.py imports work")
        
        from train import ReversalDataset, collate_fn
        print("  ✓ train.py imports work")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def run_quick_test():
    """Run a quick test of the implementation"""
    print("\nRunning quick test...")
    
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent))
        
        from implementation import Seq2SeqWithAttention
        
        # Create small model
        model = Seq2SeqWithAttention(
            src_vocab_size=50,
            trg_vocab_size=50,
            embed_size=32,
            hidden_size=64
        )
        
        # Test forward pass
        src = torch.randint(3, 50, (2, 5))
        trg = torch.randint(3, 50, (2, 4))
        src_lengths = torch.tensor([5, 3])
        
        outputs, attentions = model(src, src_lengths, trg)
        
        assert outputs.shape == (2, 3, 50), f"Wrong output shape: {outputs.shape}"
        assert attentions.shape == (2, 3, 5), f"Wrong attention shape: {attentions.shape}"
        
        print("  ✓ Model forward pass works")
        print(f"    Output shape: {outputs.shape}")
        print(f"    Attention shape: {attentions.shape}")
        
        # Test inference
        translations, attn = model.translate(src[:1], src_lengths[:1])
        print(f"  ✓ Translation works")
        print(f"    Generated {translations.shape[1]} tokens")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if requirements_path.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ])
            print("  ✓ Requirements installed")
            return True
        except subprocess.CalledProcessError:
            print("  ✗ Failed to install requirements")
            return False
    else:
        print("  ⚠ requirements.txt not found")
        return False


def main():
    """Run all setup checks"""
    print("=" * 60)
    print("Day 15: Bahdanau Attention - Setup Verification")
    print("=" * 60)
    
    results = {}
    
    results['python'] = check_python_version()
    results['pytorch'] = check_pytorch()
    results['numpy'] = check_numpy()
    results['matplotlib'] = check_matplotlib()
    results['files'] = verify_file_structure()
    
    if results['pytorch'] and results['numpy']:
        results['imports'] = test_basic_imports()
        if results['imports']:
            results['test'] = run_quick_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    all_good = True
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed and check not in ['matplotlib']:  # matplotlib is optional
            all_good = False
    
    if all_good:
        print("\n🎉 All checks passed! You're ready to start.")
        print("\nNext steps:")
        print("  1. Read README.md for paper overview")
        print("  2. Study implementation.py")
        print("  3. Start with exercises/exercise_1.py")
        print("  4. Check solutions/ if stuck")
    else:
        print("\n⚠️  Some checks failed. Run with --install to fix:")
        print(f"    python {Path(__file__).name} --install")
    
    return all_good


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup verification for Day 15")
    parser.add_argument('--install', action='store_true', 
                        help='Install required packages')
    args = parser.parse_args()
    
    if args.install:
        install_requirements()
    
    success = main()
    sys.exit(0 if success else 1)
