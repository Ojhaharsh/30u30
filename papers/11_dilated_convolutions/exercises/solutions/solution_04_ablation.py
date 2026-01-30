"""
Solution 4: Dilation Pattern Ablation
=====================================

Compare receptive fields and effects of different dilation patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def calculate_receptive_field(dilations, kernel_size=3):
    """
    Calculate total receptive field after stacking dilated convolutions.
    
    For each layer: RF increases by (k-1) * dilation
    """
    rf = 1  # Start with 1 pixel
    for d in dilations:
        rf += (kernel_size - 1) * d
    return rf


# Different dilation strategies
DILATION_PATTERNS = {
    'standard': [1, 1, 1, 1, 1, 1, 1, 1],           # All dilation=1
    'exponential': [1, 2, 4, 8, 16, 32, 64, 128],   # Powers of 2 (WaveNet)
    'linear': [1, 2, 3, 4, 5, 6, 7, 8],             # Linear growth
    'repeated': [1, 2, 4, 1, 2, 4, 1, 2],           # Repeated pattern (avoids gridding)
    'aspp': [1, 6, 12, 18],                          # DeepLab ASPP (parallel, not stacked)
    'hdc': [1, 2, 5, 1, 2, 5, 1, 2],                # Hybrid Dilated Conv
}


def compare_patterns():
    """Compare receptive fields of all patterns."""
    print("Dilation Pattern Comparison")
    print("=" * 60)
    print(f"{'Pattern':<15} {'Layers':<8} {'RF Size':<10} {'Dilations'}")
    print("-" * 60)
    
    for name, dilations in DILATION_PATTERNS.items():
        rf = calculate_receptive_field(dilations)
        dil_str = ','.join(map(str, dilations[:5])) + ('...' if len(dilations) > 5 else '')
        print(f"{name:<15} {len(dilations):<8} {rf:<10} [{dil_str}]")
    
    print("-" * 60)


def plot_rf_growth():
    """Visualize receptive field growth for different patterns."""
    plt.figure(figsize=(12, 6))
    
    for name, dilations in DILATION_PATTERNS.items():
        if name == 'aspp':  # ASPP is parallel, not sequential
            continue
            
        rfs = [1]
        for d in dilations:
            rfs.append(rfs[-1] + 2 * d)  # kernel=3, so (k-1)*d = 2*d
        
        plt.plot(range(len(rfs)), rfs, marker='o', label=name, linewidth=2)
    
    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('Receptive Field Size', fontsize=12)
    plt.title('Receptive Field Growth by Dilation Pattern', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


class DilatedStack(nn.Module):
    """Stack of dilated convolutions for testing."""
    
    def __init__(self, channels, dilations):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d)
            for d in dilations
        ])
        
    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x


def test_gridding_artifact():
    """
    Demonstrate the gridding artifact with powers of 2.
    
    When using dilations 1,2,4,8,... not all pixels contribute equally
    to the output, creating a checkerboard-like pattern.
    """
    print("\nGridding Artifact Test:")
    print("  Power-of-2 dilations can cause checkerboard artifacts")
    print("  Solution: Use non-power-of-2 (HDC) or repeated patterns")
    
    # Test with delta input (single pixel)
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 32, 32] = 1.0  # Single impulse
    
    exponential = DilatedStack(1, [1, 2, 4, 8])
    hdc = DilatedStack(1, [1, 2, 5, 1])
    
    with torch.no_grad():
        out_exp = exponential(x)
        out_hdc = hdc(x)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(x[0, 0].numpy(), cmap='gray')
    axes[0].set_title('Input (single pixel)')
    
    axes[1].imshow(out_exp[0, 0].numpy(), cmap='hot')
    axes[1].set_title('Exponential (1,2,4,8)\nCan have gridding')
    
    axes[2].imshow(out_hdc[0, 0].numpy(), cmap='hot')
    axes[2].set_title('HDC (1,2,5,1)\nMore uniform coverage')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_patterns()
    plot_rf_growth()
    test_gridding_artifact()
