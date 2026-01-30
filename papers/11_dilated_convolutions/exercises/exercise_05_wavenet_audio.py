"""
Exercise 5: WaveNet Audio
=========================

Goal: Apply dilated convolutions to audio generation.

Time: 3-4 hours
Difficulty: Very Hard ⏱️⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal convolution: output at time t only depends on inputs up to t.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # TODO 1: Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation)
        
    def forward(self, x):
        # TODO 2: Apply padding and remove future values
        # x = F.pad(x, (self.padding, 0))
        # return self.conv(x)
        pass


class WaveNetBlock(nn.Module):
    """
    Single WaveNet block with gated activation.
    """
    def __init__(self, channels, kernel_size=2, dilation=1):
        super().__init__()
        # TODO 3: Dilated convolution for filter and gate
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        
        # TODO 4: 1x1 convolutions for output
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        # TODO 5: Gated activation
        # filter_out = torch.tanh(self.filter_conv(x))
        # gate_out = torch.sigmoid(self.gate_conv(x))
        # gated = filter_out * gate_out
        
        # TODO 6: Residual and skip connections
        # residual = self.residual(gated) + x
        # skip = self.skip(gated)
        # return residual, skip
        pass


class WaveNet(nn.Module):
    """
    Simplified WaveNet for audio generation.
    """
    def __init__(self, channels=64, num_blocks=10, kernel_size=2):
        super().__init__()
        
        # TODO 7: Input projection
        self.input_conv = nn.Conv1d(1, channels, 1)
        
        # TODO 8: Stack of dilated blocks with exponentially growing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, ...
            # self.blocks.append(WaveNetBlock(channels, kernel_size, dilation))
        
        # TODO 9: Output layers
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, 256, 1)  # 256 quantization levels
        )
        
    def forward(self, x):
        # TODO 10: Forward pass collecting skip connections
        pass


if __name__ == "__main__":
    print(__doc__)
    print("WaveNet uses exponentially growing dilations for huge receptive field")
    print("10 layers: RF = 1 + 2*(1+2+4+8+...+512) = 1023 samples!")
