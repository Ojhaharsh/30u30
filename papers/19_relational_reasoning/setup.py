"""
setup.py - Diagnostic & Verification Suite for Day 19
Standardized for Day 19 of 30u30

This script performs advanced programmatic verification of the 
Relation Network's architectural properties.

Verifications:
1. Permutation Invariance (Standard).
2. Cardinality Sensitivity: Verifying that 'sum' detects set size changes 
   while 'mean' washes them out.
3. Coordinate Preservation: Verifying that (x, y) injection doesn't 
   break tensor shapes.
"""

import sys
import torch
from implementation import RelationNetwork, add_coordinates

def verify_permutation_invariance() -> bool:
    print("Verifying Relational Inductive Bias (Permutation Invariance)...")
    model = RelationNetwork(object_dim=8, output_dim=4)
    model.eval()
    
    objects = torch.randn(1, 10, 8)
    with torch.no_grad():
        out_orig = model(objects)
        
    indices = torch.randperm(10)
    shuffled_objects = objects[:, indices, :]
    with torch.no_grad():
        out_shuffled = model(shuffled_objects)
        
    diff = torch.abs(out_orig - out_shuffled).max().item()
    if diff < 1e-5:
        print(f"  [PASS] Output invariant to shuffle (diff: {diff:.2e})")
        return True
    return False

def verify_cardinality_bias() -> bool:
    """
    Verifies that 'sum' is sensitive to the number of objects, which 
    is essential for counting tasks as mentioned in Section 2.1.
    """
    print("\nVerifying Cardinality Bias (Sum vs Mean)...")
    
    # All objects are identical (ones)
    objs_5 = torch.ones(1, 5, 8)
    objs_10 = torch.ones(1, 10, 8)
    
    # 1. Test Mean Aggregator (Size-Invariant)
    model_mean = RelationNetwork(object_dim=8, aggregator='mean')
    model_mean.eval()
    with torch.no_grad():
        out_mean_5 = model_mean(objs_5)
        out_mean_10 = model_mean(objs_10)
    
    # Mean of identical objects should be nearly identical regardless of count
    mean_diff = torch.abs(out_mean_5 - out_mean_10).max().item()
    
    # 2. Test Sum Aggregator (Size-Sensitive)
    model_sum = RelationNetwork(object_dim=8, aggregator='sum')
    model_sum.eval()
    with torch.no_grad():
        out_sum_5 = model_sum(objs_5)
        out_sum_10 = model_sum(objs_10)
        
    sum_diff = torch.abs(out_sum_5 - out_sum_10).max().item()
    
    print(f"  [INFO] Mean Aggregator Diff (N=5 vs N=10): {mean_diff:.2e}")
    print(f"  [INFO] Sum Aggregator Diff  (N=5 vs N=10): {sum_diff:.2e}")
    
    if sum_diff > mean_diff * 10:
        print("  [PASS] Sum aggregator successfully preserves set-size information.")
        return True
    return False

def verify_coordinates() -> bool:
    print("\nVerifying Coordinate Injection...")
    objs = torch.randn(2, 5, 8)
    objs_coords = add_coordinates(objs)
    
    if objs_coords.shape == (2, 5, 10):
        print("  [PASS] Successfully appended (x, y) coordinates.")
        return True
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("Day 19: Relational Reasoning Masterclass")
    print("Diagnostic & Verification Suite")
    print("=" * 50)
    print()
    
    success = True
    success &= verify_permutation_invariance()
    success &= verify_cardinality_bias()
    success &= verify_coordinates()
    
    print("\n" + "-" * 50)
    if success:
        print("  [LOGICAL SMOKE TEST] PASSED")
        print("  The architecture is confirmed to be:")
        print("  1. Permutation Invariant (Order-agnostic)")
        print("  2. Cardinality Aware (Sum vs Mean bias handles counting)")
        print("  3. Spatial Ready (Coordinate preservation active)")
        print("-" * 50)
    else:
        print("  [LOGICAL SMOKE TEST] FAILED")
        print("-" * 50)
        sys.exit(1)
