"""
Exercise 5: Final Assembly - The Variational Lossy Autoencoder

In this exercise, you will combine all the pieces (Masked Convolutions, 
Gated Activations, PixelCNN, and IAF Flows) to build the complete VLAE.

The Goal:
Train the full model and verify that the 'forced information' principle 
works. You should achieve sharp reconstructions (thanks to PixelCNN) and 
a non-collapsed latent space (thanks to the restricted receptive field).
"""

import sys
import os
import torch
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import VLAE, loss_function

def train_vlae():
    """
    TODO: Train the full VLAE to solve posterior collapse.
    
    1. Use VLAE with use_flow=True.
    2. Use PixelCNNDecoder with restricted receptive field (e.g. small kernels).
    3. Train and verify that KL > 0 (e.g. around 10-20 nats for MNIST).
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement full VLAE training")

if __name__ == "__main__":
    train_vlae()
