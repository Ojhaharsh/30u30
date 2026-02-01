"""
Day 13: Environment Setup and Verification

Run this script to verify your environment is ready for Day 13.
"""

import sys
import os


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    
    major, minor = sys.version_info[:2]
    
    if major < 3 or (major == 3 and minor < 8):
        print(f"  [WARNING] Python 3.8+ recommended, you have {major}.{minor}")
    else:
        print(f"  [OK] Python {major}.{minor}")
    
    return True


def check_numpy():
    """Check NumPy installation."""
    print("Checking NumPy...")
    
    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
        
        # Quick test
        x = np.random.randn(10, 10)
        assert x.shape == (10, 10)
        print("  [OK] NumPy operations working")
        return True
        
    except ImportError:
        print("  [ERROR] NumPy not installed")
        print("  Run: pip install numpy")
        return False


def check_matplotlib():
    """Check matplotlib installation."""
    print("Checking Matplotlib...")
    
    try:
        import matplotlib
        print(f"  [OK] Matplotlib {matplotlib.__version__}")
        return True
        
    except ImportError:
        print("  [WARNING] Matplotlib not installed (optional)")
        print("  Run: pip install matplotlib")
        return True  # Optional


def check_implementation():
    """Check that core implementation works."""
    print("Checking implementation...")
    
    try:
        from implementation import (
            ScaledDotProductAttention,
            MultiHeadAttention,
            PositionalEncoding,
            EncoderBlock,
            DecoderBlock,
            Transformer
        )
        print("  [OK] All components imported")
        
        import numpy as np
        np.random.seed(42)
        
        # Test attention
        attn = ScaledDotProductAttention(dropout_p=0.0)
        attn.eval()
        
        Q = np.random.randn(2, 4, 8, 32)
        K = np.random.randn(2, 4, 8, 32)
        V = np.random.randn(2, 4, 8, 32)
        
        output = attn.forward(Q, K, V)
        assert output.shape == (2, 4, 8, 32), f"Wrong shape: {output.shape}"
        print("  [OK] ScaledDotProductAttention working")
        
        # Test multi-head attention
        mha = MultiHeadAttention(d_model=128, n_heads=4, dropout_p=0.0)
        mha.eval()
        
        x = np.random.randn(2, 10, 128)
        output = mha.forward(x, x, x)
        assert output.shape == (2, 10, 128), f"Wrong shape: {output.shape}"
        print("  [OK] MultiHeadAttention working")
        
        # Test positional encoding
        pe = PositionalEncoding(d_model=128, max_len=100)
        embeddings = np.random.randn(2, 20, 128)
        encoded = pe.forward(embeddings)
        assert encoded.shape == embeddings.shape
        print("  [OK] PositionalEncoding working")
        
        # Test encoder
        encoder = EncoderBlock(d_model=64, n_heads=4, d_ff=256, dropout_p=0.0)
        encoder.eval()
        x = np.random.randn(2, 10, 64)
        output = encoder.forward(x)
        assert output.shape == (2, 10, 64)
        print("  [OK] EncoderBlock working")
        
        # Test decoder
        decoder = DecoderBlock(d_model=64, n_heads=4, d_ff=256, dropout_p=0.0)
        decoder.eval()
        encoder_out = np.random.randn(2, 8, 64)
        decoder_in = np.random.randn(2, 6, 64)
        output = decoder.forward(decoder_in, encoder_out)
        assert output.shape == (2, 6, 64)
        print("  [OK] DecoderBlock working")
        
        # Test full transformer
        transformer = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=256,
            dropout_p=0.0
        )
        transformer.eval()
        
        src = np.random.randint(0, 100, (2, 8))
        tgt = np.random.randint(0, 100, (2, 6))
        logits = transformer.forward(src, tgt)
        assert logits.shape == (2, 6, 100)
        print("  [OK] Full Transformer working")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Implementation check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training():
    """Check that training utilities work."""
    print("Checking training utilities...")
    
    try:
        from train_minimal import (
            generate_copy_task_data,
            cross_entropy_loss,
            compute_accuracy,
            SimpleTransformer
        )
        import numpy as np
        np.random.seed(42)
        
        # Generate data
        src, tgt = generate_copy_task_data(100, 5, vocab_size=10)
        assert src.shape == (100, 5)
        assert tgt.shape == (100, 6)
        print("  [OK] Data generation working")
        
        # Create model
        model = SimpleTransformer(vocab_size=10, d_model=32, n_heads=2, n_layers=1)
        model.eval()
        print("  [OK] Model creation working")
        
        # Forward pass
        logits = model.forward(src[:2], tgt[:2, :-1])
        assert logits.shape[0] == 2
        print("  [OK] Forward pass working")
        
        # Loss
        loss = cross_entropy_loss(logits, tgt[:2, 1:])
        assert loss > 0
        print(f"  [OK] Loss computation working (loss={loss:.4f})")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Training check failed: {e}")
        return False


def check_visualization():
    """Check visualization module."""
    print("Checking visualization...")
    
    try:
        from visualization import (
            plot_attention_weights,
            plot_positional_encoding,
            plot_transformer_architecture
        )
        print("  [OK] Visualization functions imported")
        return True
        
    except Exception as e:
        print(f"  [WARNING] Visualization check failed: {e}")
        return True  # Optional


def main():
    """Run all setup checks."""
    print("=" * 60)
    print("DAY 13: ENVIRONMENT SETUP VERIFICATION")
    print("=" * 60)
    
    results = {
        'Python': check_python_version(),
        'NumPy': check_numpy(),
        'Matplotlib': check_matplotlib(),
        'Implementation': check_implementation(),
        'Training': check_training(),
        'Visualization': check_visualization(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAILED]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All checks passed! You're ready for Day 13.")
        print("\nQuick start:")
        print("  python implementation.py  # Run demo")
        print("  python train_minimal.py   # Train on copy task")
        print("  python visualization.py   # See visualizations")
    else:
        print("Some checks failed. Please fix the issues above.")
        print("Run: pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
