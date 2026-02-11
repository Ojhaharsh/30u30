"""
Exercise 2: Gated PixelCNN (Powerful Autoregressive Decoding)

In this exercise, you will implement the Gated Activation Unit and a 
PixelCNN layer as described in van den Oord et al. (2016) and used in VLAE.

The Goal:
Beyond simple masks, PixelCNNs use a gated activation (tanh * sigmoid) 
to model more complex interactions between pixels and to avoid the 
vanishing gradient problem in deep stacks.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path to import implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from implementation import PixelCNNDecoder # You can use this for reference

class GatedActivation(nn.Module):
    """
    TODO: Implement Gated Activation.
    y = tanh(W_f * x) * sigmoid(W_g * x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

class GatedPixelCNNLayer(nn.Module):
    """
    TODO: Implement a Gated PixelCNN Layer.
    This layer should combine two MaskedConv2d operations (one for vertical, one for horizontal)
    and then apply the GatedActivation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

class PixelCNN(nn.Module):
    """
    TODO: Implement a simple PixelCNN.
    
    1. Use MaskedConv2d (from your previous exercise or implementation.py).
    2. Stack multiple GatedPixelCNNLayer blocks (or just MaskedConv + ReLU).
    3. The input is [B, 1, H, W] (binary image).
    4. The output is [B, 1, H, W] (logits for pixel values).
    
    Note: For this exercise, you don't need 'conditional' input z. Just model p(x).
    """
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

def train_pixelcnn():
    # Setup MNIST data
    # Setup model
    # Training loop
    # YOUR CODE HERE
    raise NotImplementedError("Implement training loop")

if __name__ == "__main__":
    train_pixelcnn()
