"""
Variational Lossy Autoencoder (VLAE): Bridging VAEs and Autoregressive Models

Implements the core ideas from Chen et al. "Variational Lossy Autoencoder" (ICLR 2017):

1. Masked Convolutions (Type A/B) to enforce autoregressive property.
2. Gated Activation Units (tanh/sigmoid) for powerful local modeling.
3. Gated PixelCNN Decoder with a restricted receptive field.
4. Inverse Autoregressive Flow (IAF) prior to boost latent flexibility.
5. Evidence Lower Bound (ELBO) objective with bits-back interpretation.

The central demonstration: by restricting the decoder's receptive field and using
a powerful IAF prior, we prevent "posterior collapse" and force the latent space
to learn global semantic structures while the decoder handles local texture.

Reference: https://arxiv.org/abs/1611.02731
Author: 30u30 Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional


# ============================================================================
# SECTION 1: MASKED CONVOLUTIONS & PIXELCNN COMPONENTS
# ============================================================================
# The VLAE decoder is a Gated PixelCNN. Autoregression is enforced via 
# spatial masking. Type A masks the center pixel (first layer), Type B 
# includes it (subsequent layers).
# ============================================================================

class MaskedConv2d(nn.Conv2d):
    """
    Convolution with a masked weight matrix to enforce autoregressive property.
    Pixels can only see previous pixels (top-to-bottom, left-to-right).
    """
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.size()
        
        # Mask out the future (center + right, and all rows below)
        # Assuming kernel has odd size, center is at kH // 2, kW // 2
        
        # 1. Mask rows below center
        self.mask[:, :, kH // 2 + 1:, :] = 0
        # 2. Mask future columns in center row
        self.mask[:, :, kH // 2, kW // 2 + 1:] = 0
        
        # Type A: cannot see itself (center pixel masked) -> used in first layer
        # Type B: can see itself (reusing its own pixel from previous layer)
        if mask_type == 'A':
            self.mask[:, :, kH // 2, kW // 2] = 0
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class GatedActivation(nn.Module):
    """
    Gated PixelCNN activation: y = tanh(W_f * x) * sigmoid(W_g * x).
    Ref: van den Oord et al. (2016) "Conditional Image Generation with PixelCNN Decoders".
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f, g = torch.chunk(x, 2, dim=1)
        return torch.tanh(f) * torch.sigmoid(g)


class GatedPixelCNNLayer(nn.Module):
    """
    A single layer of Gated PixelCNN as described in VLAE (Section 4).
    Includes masking, gated activation, and residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 mask_type: str, conditional_dim: int = 0):
        super().__init__()
        self.mask_type = mask_type
        # Double output channels for the gated activation split (f, g)
        self.conv = MaskedConv2d(mask_type, in_channels, out_channels * 2, 
                                kernel_size, padding=kernel_size // 2)
        
        # Condition from latent z (upsampled)
        if conditional_dim > 0:
            self.cond_proj = nn.Conv2d(conditional_dim, out_channels * 2, kernel_size=1)
        else:
            self.cond_proj = None
            
        self.activation = GatedActivation()
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_in = x
        x = self.conv(x)
        
        if h is not None and self.cond_proj is not None:
            x = x + self.cond_proj(h)
            
        x = self.activation(x)
        x = self.out_proj(x)
        
        # Residual connection (only for Type B layers where channels match)
        if self.mask_type == 'B' and x.shape == x_in.shape:
            x = x + x_in
            
        return x


# ============================================================================
# SECTION 2: INVERSE AUTOREGRESSIVE FLOW (IAF)
# ============================================================================
# VLAE uses IAF for the prior p(z). This makes the prior highly flexible.
# We implement this via Masked Autoencoder for Distribution Estimation (MADE).
# ============================================================================

class MaskedLinear(nn.Linear):
    """Linear layer with a mask to preserve autoregressive property in MADE."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def set_mask(self, mask: torch.Tensor):
        self.mask.data.copy_(mask.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.
    Used as the coupling layer for IAF.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            MaskedLinear(input_dim, hidden_dim),
            nn.ReLU(),
            MaskedLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            MaskedLinear(hidden_dim, output_dim)
        )
        self.m = {}
        self.create_masks()

    def create_masks(self):
        """Standard MADE mask generation logic."""
        # Find linear layers (skipping ReLUs)
        layers = [m for m in self.net if isinstance(m, MaskedLinear)]
        L = len(layers) - 1
        
        # 1. Assign ranks to hidden units
        self.m[-1] = np.arange(self.input_dim)
        for l in range(L):
            # Each hidden unit gets a rank between 0 and D-2
            self.m[l] = np.random.randint(0, self.input_dim - 1, size=layers[l].out_features)
        
        # 2. Assign ranks to output units (must be 1:1 or N:1 mapping to input dim)
        self.m[L] = np.repeat(np.arange(self.input_dim), layers[L].out_features // self.input_dim)

        # 3. Apply masks to each layer
        layer_idx = 0
        for m in self.net:
            if isinstance(m, MaskedLinear):
                if layer_idx < L:
                    # Hidden layers: rank_prev <= rank_curr
                    mask = self.m[layer_idx-1][:, None] <= self.m[layer_idx][None, :]
                else:
                    # Output layer: rank_prev < rank_curr (STRICT for autoregression)
                    mask = self.m[layer_idx-1][:, None] < self.m[layer_idx][None, :]
                
                m.set_mask(torch.from_numpy(mask.astype(np.float32)))
                layer_idx += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IAFBlock(nn.Module):
    """
    A single step of Inverse Autoregressive Flow (IAF).
    z_out = mu(z_in) + sigma(z_in) * z_in
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.made = MADE(dim, hidden_dim, dim * 2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.made(z)
        mu, log_sigma = torch.chunk(stats, 2, dim=1)
        sigma = torch.sigmoid(log_sigma) # Use sigmoid for stability in IAF
        
        # Transformations: z_new = (z - mu) * sigma
        # In the context of VLAE prior p(z), it's used to transform epsilon -> z
        z_new = (z - mu) * sigma
        
        # Log determinant of the Jacobian
        log_det = torch.sum(torch.log(sigma), dim=1)
        
        return z_new, log_det


# ============================================================================
# SECTION 3: ENCODER & COMPLETE VLAE MODEL
# ============================================================================
# VLAE combines a standard VAE encoder, the IAF prior, and the restricted
# PixelCNN decoder.
# ============================================================================

class Encoder(nn.Module):
    """ResNet-style encoder for VLAE."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VLAE(nn.Module):
    """
    Variational Lossy Autoencoder.
    Cures posterior collapse via restricted receptive field and flow prior.
    """
    def __init__(self, input_dim: int = 1, latent_dim: int = 32, 
                 hidden_dim: int = 64, n_layers: int = 3, use_flow: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_flow = use_flow
        
        self.encoder = Encoder(input_dim, latent_dim)
        
        # Restricted Receptive Field Decoder
        # With n_layers=3 and kernel=3, the receptive field is small (7x7)
        self.initial_conv = MaskedConv2d('A', input_dim, hidden_dim, 3, padding=1)
        self.decoder_layers = nn.ModuleList([
            GatedPixelCNNLayer(hidden_dim, hidden_dim, 3, 'B', conditional_dim=latent_dim)
            for _ in range(n_layers)
        ])
        self.final_conv = nn.Conv2d(hidden_dim, input_dim, 1)
        
        # Flow Prior Steps
        if use_flow:
            self.flow_steps = nn.ModuleList([
                IAFBlock(latent_dim, 128) for _ in range(2)
            ])

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # 2. Prior Flow (for log p(z) calculation)
        log_det_total = torch.zeros(z.size(0), device=z.device)
        z_flow = z
        if self.use_flow:
            for flow in self.flow_steps:
                z_flow, log_det = flow(z_flow)
                log_det_total += log_det
        
        # 3. Decode with restricted receptive field
        # Conditionally project z onto the spatial grid
        h_latent = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        
        h = self.initial_conv(x)
        for layer in self.decoder_layers:
            h = layer(h, h_latent)
        
        logits = self.final_conv(h)
        return logits, mu, logvar, z, log_det_total


def loss_function(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                  logvar: torch.Tensor, z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    """
    VLAE ELBO Loss Function.
    Equivalent to bits-back coding cost: Cost = Reconstruction + KL
    """
    # 1. Reconstruction Loss (Binary Cross Entropy for binarized MNIST)
    recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    
    # 2. KL Divergence: log q(z|x) - log p(z)
    # log q(z|x) = sum_i log N(z_i; mu_i, var_i)
    log_q = -0.5 * torch.sum(np.log(2*np.pi) + logvar + torch.pow(z - mu, 2) / torch.exp(logvar))
    
    # log p(z) with Flow (Jacobian change of variables)
    # Start with base log p(z_base) = N(0, I)
    # log p(z) = log p(base) - log |det Jacobian|
    log_p_base = -0.5 * torch.sum(np.log(2*np.pi) + torch.pow(z, 2))
    log_p = log_p_base - log_det.sum()
    
    kl_loss = (log_q - log_p).sum()
    
    return recon_loss + kl_loss
