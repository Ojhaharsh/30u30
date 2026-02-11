"""
Exercise 1: Masked Convolutions (The Foundation of Autoregression)

In this exercise, you will implement the core mechanism that allows a 
Convolutional Neural Network (CNN) to become an autoregressive model.

The Goal:
Ensure that each pixel in an image depends only on 'previous' pixels 
(top-to-bottom, left-to-right). This prevents the model from 'cheating'
by looking at the ground-truth pixel it is supposed to predict.

Requirement:
1. Implement MaskedConv2d class.
2. Support Type A (first layer, masks center) and Type B (later layers).
"""

import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add parent directory to path to import implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MaskedConv2d(nn.Conv2d):
    """
    TODO: Implement Masked Convolution.
    
    1. It should inherit from nn.Conv2d.
    2. In __init__, create a mask buffer (register_buffer) of the same shape as weight.
    3. Initialize mask with 1s.
    4. Set the center and future pixels to 0 based on mask_type ('A' or 'B').
       - Type A: Center is 0 (cannot see itself). Used in first layer.
       - Type B: Center is 1 (can see itself). Used in subsequent layers.
       - All future pixels (bottom rows, right side of center row) must be 0.
    5. In forward, multiply weight by mask before calling super().forward().
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # YOUR CODE HERE
        raise NotImplementedError("Implement MaskedConv2d __init__")

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError("Implement MaskedConv2d forward")

def visualize_mask(mask_type):
    # This function checks your implementation visually
    try:
        conv = MaskedConv2d(mask_type, 1, 1, kernel_size=5, padding=2)
        mask = conv.mask[0, 0].cpu().numpy()
        
        plt.figure(figsize=(4, 4))
        plt.imshow(mask, cmap='gray', interpolation='nearest')
        plt.title(f'Mask Type {mask_type}')
        plt.colorbar()
        plt.show()
    except NotImplementedError as e:
        print(f"Not implemented: {e}")

if __name__ == "__main__":
    print("Visualizing Type A Mask...")
    visualize_mask('A')
    
    print("Visualizing Type B Mask...")
    visualize_mask('B')
