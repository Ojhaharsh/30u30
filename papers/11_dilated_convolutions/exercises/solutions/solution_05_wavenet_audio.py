"""
Solution 5: WaveNet-style Dilated Convolutions
==============================================

Causal dilated convolutions for audio modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: output at time t only uses inputs t, t-1, t-2, ...
    
    Key: Left-pad the input so conv doesn't see future values.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        
    def forward(self, x):
        # Pad left side only
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class WaveNetBlock(nn.Module):
    """
    Single WaveNet residual block with gated activation.
    
    Structure:
        x → dilated_conv → tanh ⊙ sigmoid → 1x1 conv → + → residual
                                              ↓
                                           skip out
    """
    
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated_conv_f = CausalConv1d(channels, channels, 2, dilation)  # Filter
        self.dilated_conv_g = CausalConv1d(channels, channels, 2, dilation)  # Gate
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        # Gated activation
        f = torch.tanh(self.dilated_conv_f(x))
        g = torch.sigmoid(self.dilated_conv_g(x))
        z = f * g
        
        # Residual and skip
        residual = self.residual_conv(z) + x
        skip = self.skip_conv(z)
        
        return residual, skip


class WaveNet(nn.Module):
    """
    Simplified WaveNet for demonstration.
    
    Uses exponentially increasing dilations: 1, 2, 4, 8, 16, ...
    This gives receptive field of 2^num_layers - 1 with only num_layers layers.
    """
    
    def __init__(self, channels=64, num_layers=10, num_classes=256):
        super().__init__()
        
        self.input_conv = nn.Conv1d(1, channels, 1)
        
        # Create blocks with exponentially growing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.blocks.append(WaveNetBlock(channels, dilation))
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, num_classes, 1)  # 8-bit mu-law = 256 levels
        )
        
        self.receptive_field = 2 ** num_layers  # RF size
        
    def forward(self, x):
        x = self.input_conv(x)
        
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        
        return self.output(skip_sum)


def calculate_wavenet_rf(num_layers, kernel_size=2):
    """Calculate WaveNet receptive field."""
    # Each layer adds (k-1) * 2^i to RF
    # For k=2: RF = sum(2^i for i in 0..n-1) = 2^n - 1
    return (kernel_size - 1) * (2 ** num_layers - 1) + 1


def visualize_receptive_field():
    """Visualize WaveNet's exponential RF growth."""
    layers = list(range(1, 11))
    rfs = [calculate_wavenet_rf(n) for n in layers]
    
    plt.figure(figsize=(10, 5))
    plt.bar(layers, rfs, color='steelblue')
    plt.xlabel('Number of Layers')
    plt.ylabel('Receptive Field (samples)')
    plt.title('WaveNet Receptive Field Growth')
    plt.yscale('log')
    
    for i, rf in enumerate(rfs):
        plt.text(layers[i], rf, str(rf), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("With 10 layers at 16kHz, RF covers:")
    print(f"  {rfs[9]} samples = {rfs[9]/16000:.3f} seconds")


def test_causality():
    """Verify causal convolution doesn't look into the future."""
    print("\nCausality Test:")
    
    conv = CausalConv1d(1, 1, kernel_size=2, dilation=4)
    
    # Create input where second half is different
    x1 = torch.zeros(1, 1, 100)
    x1[0, 0, :50] = 1.0
    
    x2 = torch.zeros(1, 1, 100)
    x2[0, 0, :50] = 1.0
    x2[0, 0, 50:] = 999.0  # Different future
    
    with torch.no_grad():
        y1 = conv(x1)
        y2 = conv(x2)
    
    # Output at position 50 should be same (only depends on past)
    diff = (y1[0, 0, :51] - y2[0, 0, :51]).abs().max().item()
    print(f"  Max difference in first 51 outputs: {diff:.6f}")
    print(f"  Causal: {'Yes ✓' if diff < 1e-6 else 'No ✗'}")


def demo():
    """Run WaveNet demo."""
    print("WaveNet Dilated Convolutions")
    print("=" * 60)
    
    model = WaveNet(channels=32, num_layers=10)
    print(f"Receptive field: {model.receptive_field} samples")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 1000)  # 1 channel, 1000 samples
    y = model(x)
    print(f"\nInput: {list(x.shape)} → Output: {list(y.shape)}")
    
    test_causality()
    visualize_receptive_field()


if __name__ == "__main__":
    demo()
