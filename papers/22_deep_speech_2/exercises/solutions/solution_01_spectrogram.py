"""
Solution 1: Spectrogram Feature Extraction
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
    """Compute a normalized log power spectrogram."""
    window_size = int(sample_rate * window_ms / 1000)
    stride = int(sample_rate * stride_ms / 1000)

    # win_length must be <= n_fft for torch.stft
    win_length = min(window_size, n_fft)

    # Pad audio if shorter than window
    if len(audio) < win_length:
        audio = F.pad(audio, (0, win_length - len(audio)))

    # Hann window
    window = torch.hann_window(win_length, device=audio.device)

    # STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=stride,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    # Power spectrum
    power = spec.abs().pow(2)

    # Log (with epsilon for numerical stability)
    log_spec = torch.log(power + 1e-10)

    # Normalize to zero mean, unit variance
    log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-10)

    return log_spec


def test_spectrogram():
    """Test the spectrogram implementation."""
    torch.manual_seed(42)

    sample_rate = 16000
    duration = 0.5
    freq = 440.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * freq * t)

    spec = compute_spectrogram(audio, sample_rate, n_fft=256)

    assert spec.dim() == 2
    n_freq = 256 // 2 + 1
    assert spec.size(0) == n_freq

    mean = spec.mean().item()
    std = spec.std().item()
    assert abs(mean) < 0.1
    assert abs(std - 1.0) < 0.2

    print(f"[OK] Spectrogram shape: {spec.shape}")
    print(f"[OK] Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_spectrogram()
