"""
Exercise 4: Architecture Ablation
=================================

Goal: Compare different dilation patterns.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn


def calculate_rf(dilations, k=3):
    """Calculate total receptive field."""
    rf = 1
    for d in dilations:
        rf += (k - 1) * d
    return rf


# Different dilation patterns to test
PATTERNS = {
    'exponential': [1, 2, 4, 8, 16, 32],     # WaveNet style
    'linear': [1, 2, 3, 4, 5, 6],            # Linear growth
    'repeated': [1, 2, 4, 1, 2, 4],          # Avoids gridding
    'aspp': [1, 6, 12, 18],                   # DeepLab style
    'hdc': [1, 2, 5, 1, 2, 5, 1, 2, 5],      # Hybrid Dilated Conv
}


def compare_patterns():
    """Compare receptive fields of different patterns."""
    print("Dilation Pattern Comparison")
    print("=" * 50)
    
    for name, dilations in PATTERNS.items():
        rf = calculate_rf(dilations)
        params = len(dilations)  # Relative
        print(f"{name:15}: RF={rf:4}, Layers={params}")


if __name__ == "__main__":
    print(__doc__)
    compare_patterns()
