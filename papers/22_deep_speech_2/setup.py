"""
setup.py — Environment verification for Day 22
================================================
Run this to check that all dependencies are installed and the
implementation loads correctly.

Usage:
    python setup.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_import(module_name: str, package_name: str = None) -> bool:
    """Try importing a module and report success/failure."""
    if package_name is None:
        package_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {package_name} ({version})")
        return True
    except ImportError:
        print(f"  [FAIL] {package_name} not found. Install with: pip install {package_name}")
        return False


def main():
    print("Day 22: Deep Speech 2 — Environment Check")
    print("=" * 50)
    print()

    print("1. Checking dependencies...")
    all_ok = True
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('numpy', 'NumPy')
    all_ok &= check_import('matplotlib', 'Matplotlib')

    print()

    if not all_ok:
        print("[FAIL] Some dependencies are missing.")
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)

    print("2. Checking implementation imports...")
    try:
        from implementation import (
            DeepSpeech2, CharEncoder, ClippedReLU,
            SequenceBatchNorm, ConvBlock, BidirectionalRNNLayer,
            compute_spectrogram, generate_synthetic_audio,
            generate_synthetic_dataset, collate_batch,
            greedy_decode, beam_search_decode,
            word_error_rate, character_error_rate,
            sortagrad_sampler, train_step, evaluate
        )
        print("  [OK] All implementation classes and functions imported")
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        sys.exit(1)

    print()
    print("3. Running quick model test...")

    import torch
    torch.manual_seed(42)

    encoder = CharEncoder('english')
    print(f"  Vocab size: {encoder.vocab_size}")

    # Generate a tiny batch
    dataset = generate_synthetic_dataset(n_samples=4, min_words=1, max_words=2)
    print(f"  Generated {len(dataset)} synthetic samples")

    batch = collate_batch(dataset, encoder, n_fft=256)
    print(f"  Batch spectrogram shape: {batch.features.shape}")
    print(f"  Batch target length: {batch.targets.shape}")

    # Build a small model
    n_freq = batch.features.size(2)
    model = DeepSpeech2(
        n_freq=n_freq,
        vocab_size=encoder.vocab_size,
        n_conv=1,
        n_rnn=2,
        rnn_hidden=64,
        rnn_type='gru'
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Forward pass
    log_probs, output_lengths = model(batch.features, batch.feature_lengths)
    print(f"  Output shape: {log_probs.shape} (time, batch, vocab)")
    print(f"  Output lengths: {output_lengths.tolist()}")

    # Greedy decode
    decoded = greedy_decode(log_probs, encoder, output_lengths)
    print(f"  Sample decode (untrained): \"{decoded[0]}\"")
    print(f"  Target:                    \"{dataset[0].transcript}\"")

    # Check ClippedReLU
    relu = ClippedReLU(20.0)
    test_input = torch.tensor([-5.0, 0.0, 10.0, 25.0, 100.0])
    test_output = relu(test_input)
    expected = torch.tensor([0.0, 0.0, 10.0, 20.0, 20.0])
    assert torch.allclose(test_output, expected), f"ClippedReLU failed: {test_output}"
    print("  [OK] ClippedReLU: min(max(x,0), 20) verified")

    # Check WER/CER
    wer = word_error_rate(["the cat"], ["a cat"])
    cer = character_error_rate(["the cat"], ["a cat"])
    print(f"  [OK] WER('the cat' vs 'a cat') = {wer:.2%}")
    print(f"  [OK] CER('the cat' vs 'a cat') = {cer:.2%}")

    # Check SortaGrad
    indices_epoch0 = sortagrad_sampler(dataset, epoch=0)
    lengths = [len(dataset[i].audio) for i in indices_epoch0]
    assert lengths == sorted(lengths), "SortaGrad failed: epoch 0 should be sorted"
    print("  [OK] SortaGrad: epoch 0 sorts by length")

    print()
    print("=" * 50)
    print("All checks passed. Ready for Day 22.")


if __name__ == '__main__':
    main()
