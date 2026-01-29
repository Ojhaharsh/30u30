"""
Exercise 4: Build ResNet-1001
=============================

Goal: Implement and train a 1001-layer ResNet.

Time: 3-4 hours
Difficulty: Very Hard ⏱️⏱️⏱️⏱️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


class PreActBottleneck(nn.Module):
    """Bottleneck block for very deep networks."""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # TODO 1: Implement pre-activation bottleneck
        # BN-ReLU-Conv1x1 → BN-ReLU-Conv3x3 → BN-ReLU-Conv1x1
        pass


def create_resnet_1001(num_classes=10):
    """
    Create ResNet-1001.
    
    Uses: 3 stages with 111 bottleneck blocks each
    111 blocks × 3 convs/block × 3 stages + 1 = 1001 layers
    """
    # TODO 2: Implement ResNet-1001
    pass


def train_with_gradient_checkpointing(model, data_loader):
    """
    Use gradient checkpointing to save memory.
    Trade compute for memory when training ultra-deep nets.
    """
    # TODO 3: Implement memory-efficient training
    pass


if __name__ == "__main__":
    print(__doc__)
    print("ResNet-1001 requires gradient checkpointing for memory")
