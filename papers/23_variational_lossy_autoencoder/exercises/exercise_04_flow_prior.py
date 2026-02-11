"""
Exercise 4: Normalizing Flows (Breaking the Gaussian Bottleneck)

In this exercise, you will implement the Inverse Autoregressive Flow (IAF)
as described in Kingma et al. (2016) and used in the VLAE paper.

The Goal:
Standard VAEs use a simple Gaussian prior $p(z) = \mathcal{N}(0, I)$. 
A Flow Prior allows the model to learn a much more complex and flexible
latent distribution, reducing the information cost in the ELBO.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleFlow(nn.Module):
    """
    TODO: Implement a Planar Flow step.
    
    z' = z + u * h(w^T * z + b)
    
    1. Define parameters u, w, b.
    2. Implement forward pass.
    3. Compute log_det_jacobian.
       - formula: log|1 + u^T * psi(z)|
       - psi(z) = h'(linear) * w
    """
    def __init__(self, dim):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, z):
        # YOUR CODE HERE
        pass

def test_flow_prior():
    # 1. Instantiate Flow
    # 2. Sample from N(0, I) and pass through flow
    # 3. Visualize the transformation (e.g. 2D scatter plot)
    # YOUR CODE HERE
    raise NotImplementedError("Implement flow test")

if __name__ == "__main__":
    test_flow_prior()
