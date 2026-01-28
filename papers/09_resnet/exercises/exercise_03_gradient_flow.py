"""
Exercise 3: Gradient Flow Analysis
===================================

Goal: Visualize how gradients flow through ResNet vs plain network.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def analyze_gradient_flow(model, x):
    """
    Record gradient magnitudes at each layer during backward pass.
    """
    gradients = []
    
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].abs().mean().item())
    
    # TODO 1: Register hooks on all conv layers
    handles = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # TODO: h = layer.register_backward_hook(hook)
            # handles.append(h)
            pass
    
    # TODO 2: Forward + backward
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    return gradients


def plot_gradient_flow(plain_grads, resnet_grads):
    """Plot gradient magnitude comparison."""
    plt.figure(figsize=(10, 6))
    # TODO 3: Plot both gradient profiles
    # Expect: ResNet maintains gradients, Plain network vanishes
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    # Run analysis
