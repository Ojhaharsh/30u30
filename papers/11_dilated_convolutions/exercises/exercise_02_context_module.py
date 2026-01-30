"""
Exercise 2: Build Context Module
================================

Goal: Implement a multi-scale context aggregation module.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn


class DilatedContextModule(nn.Module):
    """
    Multi-scale context module using parallel dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super().__init__()
        
        # TODO 1: Create parallel dilated convolutions
        self.branches = nn.ModuleList()
        for d in dilations:
            # TODO: Each branch has dilation d
            # branch = nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d)
            pass
        
        # TODO 2: Fusion layer
        self.fusion = None  # TODO: nn.Conv2d(out_channels * len(dilations), out_channels, 1)
        
    def forward(self, x):
        # TODO 3: Forward through all branches
        branch_outputs = []
        for branch in self.branches:
            # branch_outputs.append(branch(x))
            pass
        
        # TODO 4: Concatenate and fuse
        # concat = torch.cat(branch_outputs, dim=1)
        # return self.fusion(concat)
        pass


if __name__ == "__main__":
    print(__doc__)
