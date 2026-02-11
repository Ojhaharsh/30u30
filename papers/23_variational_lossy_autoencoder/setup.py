"""
setup.py - Verification Suite for Day 23 (VLAE)

This script performs unit tests on the core components of the 
Variational Lossy Autoencoder:
1. Masked Convolutions (Type A/B logic)
2. MADE (Autoregressive verification)
3. VLAE (Forward pass & Loss stability)

Usage:
    python setup.py
"""

import torch
import unittest
import numpy as np
from implementation import MaskedConv2d, VLAE, GatedActivation, MADE, loss_function

class TestVLAE(unittest.TestCase):
    def test_masked_conv_type_a(self):
        # Type A should have 0 at the center
        conv = MaskedConv2d('A', 1, 1, 3, padding=1)
        center_val = conv.mask[0, 0, 1, 1]
        self.assertEqual(center_val, 0, "Type A mask center should be 0")
        
    def test_made_autoregressive(self):
        # Simple test to ensure MADE output for dim i depends only on <i
        dim = 10
        model = MADE(dim, 32, dim)
        x = torch.randn(1, dim, requires_grad=True)
        out = model(x)
        
        for i in range(dim):
            loss = out[0, i]
            loss.backward(retain_graph=True)
            # Gradient for x[j] where j >= i should be 0
            grads = x.grad.data.clone()
            self.assertTrue(torch.all(grads[0, i:] == 0), f"MADE violation at dim {i}")
            x.grad.zero_()

    def test_vlae_forward(self):
        # Test with and without flow
        for use_flow in [True, False]:
            model = VLAE(input_dim=1, latent_dim=10, use_flow=use_flow)
            x = torch.randn(2, 1, 28, 28)
            logits, mu, logvar, z, log_det = model(x)
            self.assertEqual(logits.shape, x.shape)
            self.assertEqual(z.shape, (2, 10))
            
            # Loss test
            loss = loss_function(logits, (x > 0).float(), mu, logvar, z, log_det)
            self.assertTrue(torch.isfinite(loss))

if __name__ == '__main__':
    unittest.main()
