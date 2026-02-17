"""
exercise_04_parameter_count.py - Count Parameters in CNN Architectures

Difficulty: Medium (3/5)

Task: Build a parameter counter that computes the total number of
learnable parameters for a given CNN architecture. This exercise uses
the VGGNet-16 architecture from CS231n as the primary test case.

Key insight from CS231n: Most parameters are in the FC layers
(VGGNet's first FC layer alone has 102M of the 138M total), but most
memory usage is in the early CONV layers.

Formula for CONV parameters: (F * F * D_in + 1) * K
(+1 for the bias per filter)

Formula for FC parameters: (D_in + 1) * D_out
(+1 for the bias per output neuron)

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""


def count_parameters(architecture):
    """
    Count learnable parameters for each layer in a CNN.

    YOUR TASK: Fill in this function.

    Args:
        architecture: List of layer specs, each a dict:
            {'type': 'conv', 'in_depth': D, 'filters': K, 'filter_size': F}
            {'type': 'pool'}  # No parameters
            {'type': 'fc', 'in_size': D_in, 'out_size': D_out}

    Returns:
        results: List of (layer_name, param_count) tuples
        total: Total parameter count

    Hints:
        - CONV: (F * F * D_in + 1) * K  (weights + 1 bias per filter)
        - POOL: 0 parameters (fixed function)
        - FC:   (D_in + 1) * D_out      (weights + 1 bias per output)
    """
    raise NotImplementedError("Implement parameter counting")

    # results = []
    # total = 0
    # for layer in architecture:
    #     if layer['type'] == 'conv':
    #         params = ???
    #         results.append((f"CONV{layer['filter_size']}-{layer['filters']}", params))
    #     elif layer['type'] == 'pool':
    #         results.append(("POOL", 0))
    #     elif layer['type'] == 'fc':
    #         params = ???
    #         results.append((f"FC-{layer['out_size']}", params))
    #     total += params if layer['type'] != 'pool' else 0
    # return results, total


def check():
    """Test against CS231n's VGGNet-16 parameter counts."""

    # Test 1: Single CONV layer
    arch = [
        {'type': 'conv', 'in_depth': 3, 'filters': 64, 'filter_size': 3},
    ]
    results, total = count_parameters(arch)
    # (3*3*3 + 1) * 64 = 28 * 64 = 1792
    assert total == 1792, f"Test 1 FAILED: expected 1792, got {total}"
    print("[OK] Test 1: Conv3-64 with RGB input = 1,792 params")

    # Test 2: FC layer
    arch = [
        {'type': 'fc', 'in_size': 4096, 'out_size': 4096},
    ]
    results, total = count_parameters(arch)
    # (4096 + 1) * 4096 = 16,781,312
    assert total == (4096 + 1) * 4096, f"Test 2 FAILED: expected {(4096+1)*4096}, got {total}"
    print("[OK] Test 2: FC 4096->4096 = 16,781,312 params")

    # Test 3: Pool has zero parameters
    arch = [
        {'type': 'pool'},
    ]
    results, total = count_parameters(arch)
    assert total == 0, f"Test 3 FAILED: pool should have 0 params, got {total}"
    print("[OK] Test 3: Pool layer has 0 parameters")

    # Test 4: VGGNet-16 first CONV layer (from CS231n)
    # "CONV3-64: weights: (3*3*3)*64 = 1,728"
    # Note: CS231n reports 1,728 without counting biases.
    # With biases: (3*3*3 + 1) * 64 = 1,792
    # We count with biases (standard practice).
    arch = [
        {'type': 'conv', 'in_depth': 3, 'filters': 64, 'filter_size': 3},
    ]
    results, total = count_parameters(arch)
    print(f"[OK] Test 4: VGG first CONV: {total} params (CS231n reports 1,728 weights + 64 biases)")

    # Test 5: VGGNet-16 first FC (the big one)
    # Input: 7*7*512 = 25088, output: 4096
    arch = [
        {'type': 'fc', 'in_size': 7*7*512, 'out_size': 4096},
    ]
    results, total = count_parameters(arch)
    expected = (7*7*512 + 1) * 4096
    assert total == expected, f"Test 5 FAILED: expected {expected}, got {total}"
    print(f"[OK] Test 5: VGG first FC = {total:,} params (74% of total VGG params)")

    # Test 6: Full VGG-16 (simplified — all CONV and FC layers)
    vgg16 = [
        {'type': 'conv', 'in_depth': 3, 'filters': 64, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 64, 'filters': 64, 'filter_size': 3},
        {'type': 'pool'},
        {'type': 'conv', 'in_depth': 64, 'filters': 128, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 128, 'filters': 128, 'filter_size': 3},
        {'type': 'pool'},
        {'type': 'conv', 'in_depth': 128, 'filters': 256, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 256, 'filters': 256, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 256, 'filters': 256, 'filter_size': 3},
        {'type': 'pool'},
        {'type': 'conv', 'in_depth': 256, 'filters': 512, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 512, 'filters': 512, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 512, 'filters': 512, 'filter_size': 3},
        {'type': 'pool'},
        {'type': 'conv', 'in_depth': 512, 'filters': 512, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 512, 'filters': 512, 'filter_size': 3},
        {'type': 'conv', 'in_depth': 512, 'filters': 512, 'filter_size': 3},
        {'type': 'pool'},
        {'type': 'fc', 'in_size': 7*7*512, 'out_size': 4096},
        {'type': 'fc', 'in_size': 4096, 'out_size': 4096},
        {'type': 'fc', 'in_size': 4096, 'out_size': 1000},
    ]
    results, total = count_parameters(vgg16)
    print(f"\n[OK] Test 6: Full VGG-16 = {total:,} params")

    # Print breakdown
    print("\nVGG-16 Parameter Breakdown:")
    for name, count in results:
        if count > 0:
            pct = count / total * 100
            print(f"  {name:15s} {count:>12,}  ({pct:5.1f}%)")

    conv_total = sum(c for name, c in results if name.startswith('CONV'))
    fc_total = sum(c for name, c in results if name.startswith('FC'))
    print(f"\n  CONV total:    {conv_total:>12,}  ({conv_total/total*100:.1f}%)")
    print(f"  FC total:      {fc_total:>12,}  ({fc_total/total*100:.1f}%)")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
