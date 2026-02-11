"""
Exercise 3: Reproducing Posterior Collapse

In this exercise, you will witness the very problem VLAE was designed to 
solve. You will build a 'too powerful' VAE and watch the KL divergence 
sink to zero during training.

The Goal:
Understand that increasing model capacity (more layers, larger filters)
is not always better if it leads to the latent space being ignored.
"""

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import VLAE, loss_function

def train_collapsing_vae():
    """
    TODO: Demonstrate Posterior Collapse.
    
    1. Initialize VLAE with a POWERFUL decoder (PixelCNN with large receptive field).
       - You can modify the PixelCNNDecoder parameters or implementation.py to allow custom receptive field.
       - Or just use the default one but with a weak prior (Standard Gaussian, use_flow=False).
       
    2. Train on MNIST.
    3. Monitor the KL Divergence term.
    
    Expected Result: KL should drop to near 0, meaning the model ignores z.
    """
    
    # Setup Data
    # Setup Model (use_flow=False)
    # Training Loop
    # Record KL divergence
    # Plot KL over time
    
    # YOUR CODE HERE
    raise NotImplementedError("Implement collapse demo")

if __name__ == "__main__":
    train_collapsing_vae()
