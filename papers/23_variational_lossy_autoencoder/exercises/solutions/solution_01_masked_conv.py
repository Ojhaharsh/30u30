"""
Solution 01: Masked Convolutions

Implementation of Type A and Type B masks to enforce autoregression in 2D space.
Used in Gated PixelCNN and VLAE decoders.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import MaskedConv2d

def visualize_mask(mask_type):
    conv = MaskedConv2d(mask_type, 1, 1, kernel_size=5, padding=2)
    mask = conv.mask[0, 0].cpu().numpy()
    
    plt.figure(figsize=(4, 4))
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.title(f'Mask Type {mask_type}')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    print("Visualizing Type A Mask (Center is 0)...")
    visualize_mask('A')
    
    print("Visualizing Type B Mask (Center is 1)...")
    visualize_mask('B')
    
    print("\nTest passed if Type A has black center pixel and Type B has white center pixel.")
    # Also verify programmatically
    conv_a = MaskedConv2d('A', 1, 1, 5, padding=2)
    center_val_a = conv_a.mask[0, 0, 2, 2]
    assert center_val_a == 0, "Type A mask center should be 0"
    
    conv_b = MaskedConv2d('B', 1, 1, 5, padding=2)
    center_val_b = conv_b.mask[0, 0, 2, 2]
    assert center_val_b == 1, "Type B mask center should be 1"
    print("Assertions passed!")
