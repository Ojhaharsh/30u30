"""
Exercise 4: Depth Scaling
=========================

Goal: Train ResNets of various depths and compare.

Time: 2-3 hours
Difficulty: Hard ⏱️⏱️⏱️
"""

import torch
import torch.nn as nn


def create_resnet(depth, num_classes=10):
    """
    Create ResNet of specified depth.
    
    Depth mapping:
    - 18: [2, 2, 2, 2] blocks
    - 34: [3, 4, 6, 3] blocks
    - 50: [3, 4, 6, 3] bottleneck blocks
    """
    # TODO 1: Map depth to block configuration
    configs = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
    }
    
    # TODO 2: Build ResNet with given config
    pass


def compare_depths():
    """Train ResNet-18, 34, 50 and compare."""
    depths = [18, 34, 50]
    results = {}
    
    for d in depths:
        # TODO 3: Train each model and record metrics
        pass
    
    # TODO 4: Plot comparison


if __name__ == "__main__":
    print(__doc__)
    # compare_depths()
