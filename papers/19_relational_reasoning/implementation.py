"""
Relation Network (RN) Implementation: Relational Reasoning Module
Standardized for Day 19 of 30u30

This module implements the Relation Network (RN) as described in Santoro et al. 
(2017). The RN is a neural architecture with a specific inductive bias for 
relational reasoning: explicitly computing non-linear functions over every 
possible pair of objects in a set.

Expert-Grade Features:
1. Object Pair Generation: N^2 pairing via broadcasting (Section 2.1).
2. Coordinate-Aware Encoding: Optional (x, y) appending for spatial tasks (Section 3.1).
3. Variant Aggregators: Support for 'sum', 'mean', 'max' to demonstrate counting bias.
4. Relational Bottleneck (g_theta): Shared MLP for pairwise logic.
5. Global Reasoning (f_phi): MLP for final task prediction.

References:
- Santoro et al. (2017) - "A Simple Neural Network Module for Relational Reasoning"
  https://arxiv.org/abs/1706.01427

Author: 30u30 Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class RelationNetwork(nn.Module):
    """
    The Relation Network (RN) module.
    
    Architecture (from Section 2.1 of the paper):
    RN(O) = f_phi(AGGREGATE_{i,j} g_theta(o_i, o_j, q))
    
    Standard RNs use 'sum' as the aggregator because it preserves cardinality
    information (counting), which 'mean' and 'max' often destroy.
    """
    
    def __init__(
        self, 
        object_dim: int, 
        relation_dim: int = 256, 
        output_dim: int = 10, 
        question_dim: Optional[int] = None,
        aggregator: str = 'sum',
        use_coordinates: bool = False,
        dropout_p: float = 0.5
    ):
        """
        Initializes the Relation Network with research-grade flexibility.

        Args:
            object_dim: Dimension of each input object vector.
            relation_dim: Hidden dimension for the g_theta and f_phi MLPs.
            output_dim: Dimension of the final output (e.g., number of classes).
            question_dim: (Optional) Dimension of the question/context vector.
            aggregator: Strategy for combining relations ('sum', 'mean', 'max').
            use_coordinates: If True, expects (batch, N, D+2) where last 2 are (x,y).
            dropout_p: Dropout probability applied in f_phi (default 0.5 from paper).
        """
        super(RelationNetwork, self).__init__()
        
        self.aggregator = aggregator.lower()
        self.use_coords = use_coordinates
        
        # g_theta: Shared MLP applied to each pair (o_i, o_j)
        # Input size: 2 * object_dimension + (optional) question_dimension
        g_input_dim = 2 * object_dim
        if question_dim is not None:
            g_input_dim += question_dim
            
        self.g_theta = nn.Sequential(
            nn.Linear(g_input_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU()
        )
        
        # f_phi: Global MLP applied to the aggregated g_theta outputs
        self.f_phi = nn.Sequential(
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(relation_dim // 2, output_dim)
        )

    def generate_pairs(
        self, 
        objects: torch.Tensor, 
        question: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates N^2 pairs from the set of objects using broadcasting.

        Args:
            objects: Input tensor of shape (batch, num_objects, object_dim)
            question: Optional context tensor of shape (batch, question_dim)

        Returns:
            pairs: Concatenated pairs tensor of shape (batch, num_objects^2, pair_dim)
        """
        batch_size, num_objects, object_dim = objects.size()
        
        # 1. Expand objects for Cartesian product
        # o_i shape: (B, N, N, D)
        o_i = objects.unsqueeze(2).repeat(1, 1, num_objects, 1)
        # o_j shape: (B, N, N, D)
        o_j = objects.unsqueeze(1).repeat(1, num_objects, 1, 1)
        
        # 2. Concatenate pairs (o_i, o_j)
        pairs = torch.cat([o_i, o_j], dim=-1)
        
        # 3. Add question context if present (appended to every pair)
        if question is not None:
            # q_expanded shape: (B, N, N, Q_dim)
            q_expanded = question.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_objects, 1)
            pairs = torch.cat([pairs, q_expanded], dim=-1)
            
        # Reshape to (B, N^2, pair_dim) for efficient batch processing in g_theta
        return pairs.view(batch_size, num_objects * num_objects, -1)

    def aggregate(self, g_out: torch.Tensor) -> torch.Tensor:
        """
        Aggregates pairwise relations using the specified strategy.
        
        Args:
            g_out: (batch, num_pairs, relation_dim)
            
        Returns:
            aggregated: (batch, relation_dim)
        """
        if self.aggregator == 'sum':
            return g_out.sum(dim=1)
        elif self.aggregator == 'mean':
            return g_out.mean(dim=1)
        elif self.aggregator == 'max':
            return g_out.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

    def forward(
        self, 
        objects: torch.Tensor, 
        question: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Relation Network.

        Args:
            objects: (batch, num_objects, object_dim)
            question: Optional (batch, question_dim)

        Returns:
            logits: (batch, output_dim)
        """
        # Step 1: Generate N^2 pairwise object pairings
        pairs = self.generate_pairs(objects, question)
        
        # Step 2: Apply g_theta to each pair (shared weights)
        batch_size, num_pairs, pair_dim = pairs.size()
        # Flatten temporal/set dimension into batch dimension for efficiency
        pairs_flat = pairs.view(batch_size * num_pairs, pair_dim)
        g_out_flat = self.g_theta(pairs_flat)
        
        # Reshape back: (batch, num_pairs, relation_dim)
        g_out = g_out_flat.view(batch_size, num_pairs, -1)
        
        # Step 3: Global Aggregation
        # Summation is the default (preserves cardinality)
        aggregated = self.aggregate(g_out)
        
        # Step 4: Reasoning via f_phi
        output = self.f_phi(aggregated)
        
        return output


def add_coordinates(objects: torch.Tensor) -> torch.Tensor:
    """
    Appends normalized (x, y) coordinates to each object vector.
    
    Critical for spatial reasoning (CLEVR-style, Section 3.1).
    Without these, the model cannot distinguish between identical items.
    
    Args:
        objects: (batch, N, D)
    
    Returns:
        objects_with_coords: (batch, N, D+2)
    """
    batch_size, n_objects, _ = objects.size()
    
    # Create coordinate grid in [-1, 1]
    coords = torch.linspace(-1, 1, n_objects).to(objects.device)
    # For a 1D sequence or pre-extracted set:
    x_coords = coords.view(1, n_objects, 1).repeat(batch_size, 1, 1)
    y_coords = torch.zeros_like(x_coords) 
    
    return torch.cat([objects, x_coords, y_coords], dim=-1)


def feature_map_to_objects(f_map: torch.Tensor) -> torch.Tensor:
    """
    Converts a CNN feature map into a set of 'objects' (Section 3.1).
    
    In the Santoro paper, each pixel/cell in the final k x k feature map 
    is treated as an individual object.
    
    Args:
        f_map: CNN features of shape (batch, channels, k, k)
        
    Returns:
        objects: Set of objects of shape (batch, k*k, channels)
    """
    batch, channels, k, _ = f_map.size()
    # 1. Permute to (batch, k, k, channels)
    # 2. Flatten spatial dimensions into a 'set' of size k*k
    objects = f_map.permute(0, 2, 3, 1).contiguous()
    return objects.view(batch, k * k, channels)


if __name__ == "__main__":
    # Deepening Smoke Test: Comparing sum vs mean aggregators
    obj_num, obj_d = 5, 10
    
    rn_sum = RelationNetwork(object_dim=obj_d, aggregator='sum')
    rn_mean = RelationNetwork(object_dim=obj_d, aggregator='mean')
    
    dummy_objs = torch.ones(1, obj_num, obj_d) # All objects identical
    
    out_sum = rn_sum(dummy_objs)
    out_mean = rn_mean(dummy_objs)
    
    print(f"RN (Sum) Output Magnitude: {torch.norm(out_sum).item():.2f}")
    print(f"RN (Mean) Output Magnitude: {torch.norm(out_mean).item():.2f}")
    print("\nNote: Sum magnitude should be larger as it scales with N^2 pairs.")
