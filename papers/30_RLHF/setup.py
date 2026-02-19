"""
setup.py - Verify environment for Day 30: RLHF

Run this to check that all dependencies are available and the implementation is valid:
    python setup.py

Reference: Christiano et al. (2017) - https://arxiv.org/abs/1706.03741
"""

import sys
import importlib

def check_dependencies():
    """Check that required packages are installed."""
    required = {
        'numpy': 'numpy',
        'torch': 'torch',
        'gymnasium': 'gymnasium',
        'matplotlib': 'matplotlib',
    }
    
    missing = []
    print("Checking dependencies...")
    for name, import_name in required.items():
        try:
            importlib.import_module(import_name)
            print(f"  [OK] {name}")
        except ImportError:
            try:
                # Fallback for gym if gymnasium is missing (legacy support)
                if name == 'gymnasium':
                    importlib.import_module('gym')
                    print(f"  [OK] gym (using legacy gym instead of gymnasium)")
                    continue
            except ImportError:
                pass
                
            print(f"  [FAIL] {name} -- install with: pip install {name}")
            missing.append(name)
            
    return missing

def check_implementation():
    """Verify core implementation imports work."""
    print("\nChecking implementation...")
    try:
        from implementation import RewardModel, RLHF_Trainer, SyntheticOracle
        print("  [OK] implementation.py imports")
    except ImportError as e:
        print(f"  [FAIL] implementation.py import error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] implementation.py error: {e}")
        return False
        
    # Quick smoke test
    try:
        print("  Running smoke test...")
        import torch
        
        # 1. Test Reward Model shape
        obs_dim = 4
        rm = RewardModel(obs_dim)
        dummy_obs = torch.randn(1, obs_dim)
        reward = rm(dummy_obs)
        assert reward.shape == (1, 1), f"RewardModel output shape mismatch: {reward.shape}"
        print("  [OK] RewardModel forward pass")
        
        # 2. Test Oracle
        class MockEnv:
            def reset(self): return [0]*4, {}
            def step(self, a): return [0]*4, 1.0, False, False, {}
            
        oracle = SyntheticOracle(MockEnv())
        s1 = [1.0, 1.0]
        s2 = [0.0, 0.0]
        choice = oracle.query(s1, s2)
        assert choice == 0, f"Oracle logic failed (expected 0 for better reward, got {choice})"
        print("  [OK] SyntheticOracle logic")
        
    except Exception as e:
        print(f"  [FAIL] Smoke test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Day 30: Deep Reinforcement Learning from Human Feedback")
    print("Setup Verification")
    print("=" * 50)
    print()
    
    missing = check_dependencies()
    
    if missing:
        print("\nMissing critical packages. Please install them:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)
        
    if check_implementation():
        print("\nAll checks passed. Ready to go.")
        print()
        print("Quick start:")
        print("  python train_minimal.py --env CartPole-v1 --steps 5000")
        print()
        print("Or run the visualization:")
        print("  python visualization.py")
    else:
        print("\nSome checks failed. See errors above.")
        sys.exit(1)
