"""
Solution 2: Building the Relation Function (g_theta)
Standardized for Day 19 of 30u30

Based on: Santoro et al. (2017)
"""

import torch
import torch.nn as nn

class RelationFunction(nn.Module):
    """
    A 4-layer MLP as described in the RN paper.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
