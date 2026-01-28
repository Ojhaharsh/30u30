"""
Solution 3: Gradient Flow Analysis
===================================

Visualize gradient magnitudes through deep networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class PlainBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


def build_network(block_type, num_blocks, channels=64):
    """Build plain or residual network."""
    blocks = [block_type(channels) for _ in range(num_blocks)]
    return nn.Sequential(
        nn.Conv2d(3, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        *blocks
    )


def get_gradient_norms(model, x):
    """Get gradient norms at each layer during backward pass."""
    # Forward
    x.requires_grad_(True)
    y = model(x)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    # Collect gradient norms for each conv layer
    grads = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.weight.grad is not None:
                grads.append(module.weight.grad.norm().item())
    
    return grads


def analyze_gradient_flow():
    """Compare gradient flow in plain vs residual networks."""
    print("Gradient Flow Analysis")
    print("=" * 60)
    
    depths = [10, 20, 30, 40, 50]
    channels = 64
    
    plain_grads = []
    res_grads = []
    
    for num_blocks in depths:
        # Plain network
        plain_net = build_network(PlainBlock, num_blocks, channels)
        x = torch.randn(1, 3, 32, 32)
        grads = get_gradient_norms(plain_net, x.clone())
        plain_grads.append(np.mean(grads) if grads else 0)
        
        # Residual network
        res_net = build_network(ResBlock, num_blocks, channels)
        grads = get_gradient_norms(res_net, x.clone())
        res_grads.append(np.mean(grads))
        
        print(f"Depth {num_blocks*2}: Plain grad={plain_grads[-1]:.6f}, ResNet grad={res_grads[-1]:.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean gradient vs depth
    axes[0].semilogy([d*2 for d in depths], plain_grads, 'b-o', label='Plain', linewidth=2)
    axes[0].semilogy([d*2 for d in depths], res_grads, 'r-o', label='ResNet', linewidth=2)
    axes[0].set_xlabel('Network Depth (conv layers)')
    axes[0].set_ylabel('Mean Gradient Norm (log scale)')
    axes[0].set_title('Gradient Magnitude vs Depth')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Per-layer gradient for deep network
    num_blocks = 30
    
    plain_net = build_network(PlainBlock, num_blocks, channels)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    plain_per_layer = get_gradient_norms(plain_net, x.clone())
    
    res_net = build_network(ResBlock, num_blocks, channels)
    res_per_layer = get_gradient_norms(res_net, x.clone())
    
    axes[1].semilogy(range(len(plain_per_layer)), plain_per_layer, 'b-', alpha=0.7, label='Plain')
    axes[1].semilogy(range(len(res_per_layer)), res_per_layer, 'r-', alpha=0.7, label='ResNet')
    axes[1].set_xlabel('Layer Index (from input)')
    axes[1].set_ylabel('Gradient Norm (log scale)')
    axes[1].set_title(f'Per-Layer Gradients ({num_blocks*2}-layer networks)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Observations:")
    print("  - Plain network: Gradients vanish exponentially with depth")
    print("  - ResNet: Skip connections maintain gradient magnitude")
    print("  - This enables training of very deep networks")


if __name__ == "__main__":
    analyze_gradient_flow()
