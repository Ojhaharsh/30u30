"""
Exercise 1: Spectrogram Feature Extraction
Difficulty: Easy (2/5) | Time: 30 min

Deep Speech 2 takes log power spectrograms as input (Section 3.1).
Your job: implement the function that converts a raw audio waveform
into a normalized log power spectrogram.

Steps:
    1. Apply a Hann window to overlapping frames
    2. Compute the Short-Time Fourier Transform (STFT)
    3. Compute the power spectrum (magnitude squared)
    4. Take the log (with epsilon for numerical stability)
    5. Normalize to zero mean and unit variance

The paper uses 20ms windows with 10ms stride.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    window_ms: float = 20.0,
    stride_ms: float = 10.0,
    n_fft: int = 512
) -> torch.Tensor:
    """Compute a normalized log power spectrogram.

    Args:
        audio: Raw waveform, shape (num_samples,)
        sample_rate: Sample rate in Hz
        window_ms: Window size in milliseconds (paper uses 20ms)
        stride_ms: Stride in milliseconds (paper uses 10ms)
        n_fft: FFT size (determines number of frequency bins)

    Returns:
        Normalized log power spectrogram, shape (n_freq, n_time)
        where n_freq = n_fft // 2 + 1

    Hints:
        - window_size = int(sample_rate * window_ms / 1000)
        - stride = int(sample_rate * stride_ms / 1000)
        - Use torch.hann_window for the window function
        - Use torch.stft with return_complex=True
        - Power = |STFT|^2
        - log_spec = log(power + 1e-10) to avoid log(0)
        - Normalize: (log_spec - mean) / (std + 1e-10)
    """
    # TODO: Calculate window_size and stride from sample_rate and milliseconds
    window_size = None  # YOUR CODE HERE
    stride = None  # YOUR CODE HERE

    # TODO: Pad audio if shorter than window_size
    # YOUR CODE HERE

    # TODO: Create Hann window
    window = None  # YOUR CODE HERE

    # TODO: Compute STFT
    spec = None  # YOUR CODE HERE

    # TODO: Compute power spectrum (magnitude squared)
    power = None  # YOUR CODE HERE

    # TODO: Take log (add small epsilon for numerical stability)
    log_spec = None  # YOUR CODE HERE

    # TODO: Normalize to zero mean, unit variance
    log_spec = None  # YOUR CODE HERE

    return log_spec


def test_spectrogram():
    """Test your spectrogram implementation."""
    torch.manual_seed(42)

    # Generate a simple sine wave
    sample_rate = 16000
    duration = 0.5  # seconds
    freq = 440.0  # A4 note
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * freq * t)

    spec = compute_spectrogram(audio, sample_rate, n_fft=256)

    # Check shape
    assert spec is not None, "compute_spectrogram returned None"
    assert spec.dim() == 2, f"Expected 2D tensor, got {spec.dim()}D"
    n_freq = 256 // 2 + 1
    assert spec.size(0) == n_freq, f"Expected {n_freq} freq bins, got {spec.size(0)}"
    assert spec.size(1) > 0, "Spectrogram has no time frames"

    # Check normalization (approximately zero mean, unit variance)
    mean = spec.mean().item()
    std = spec.std().item()
    assert abs(mean) < 0.1, f"Mean should be ~0, got {mean:.4f}"
    assert abs(std - 1.0) < 0.2, f"Std should be ~1, got {std:.4f}"

    print(f"[OK] Spectrogram shape: {spec.shape}")
    print(f"[OK] Mean: {mean:.4f} (should be ~0)")
    print(f"[OK] Std: {std:.4f} (should be ~1)")
    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_spectrogram()
