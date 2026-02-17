"""
exercise_03_output_sizes.py - Compute CNN Output Dimensions

Difficulty: Easy (2/5)

Task: Given a CNN architecture specification, compute the output spatial
dimensions at each layer. This is pure math — no NumPy needed.

The formulas (from CS231n):
    CONV: output = (W - F + 2P) / S + 1
    POOL: output = (W - F) / S + 1

Getting these right is essential. If your dimensions don't "work out"
(produce non-integer results), the architecture is invalid.

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""


def compute_output_size(input_size, layers):
    """
    Compute spatial dimensions at each layer of a CNN.

    YOUR TASK: Fill in this function.

    Args:
        input_size: Starting spatial dimension (assumes square input)
        layers: List of layer descriptions, each a dict:
            {'type': 'conv', 'filter': F, 'stride': S, 'pad': P, 'filters': K}
            {'type': 'pool', 'filter': F, 'stride': S}

    Returns:
        sizes: List of (layer_name, spatial_size, depth) tuples

    Hints:
        - CONV: size = (size - F + 2P) / S + 1, depth = K
        - POOL: size = (size - F) / S + 1, depth unchanged
        - Check that each result is an integer (no fractional neurons)
    """
    raise NotImplementedError("Implement output size computation")

    # sizes = [("Input", input_size, 3)]
    # current_size = input_size
    # current_depth = 3
    #
    # for i, layer in enumerate(layers):
    #     if layer['type'] == 'conv':
    #         new_size = ???
    #         current_depth = layer['filters']
    #     elif layer['type'] == 'pool':
    #         new_size = ???
    #
    #     # Check for validity
    #     assert new_size == int(new_size), f"Layer {i}: fractional output {new_size}"
    #     current_size = int(new_size)
    #     sizes.append((f"Layer {i+1} ({layer['type']})", current_size, current_depth))
    #
    # return sizes


def check():
    """Test your implementation against CS231n examples."""

    # Test 1: Simple case — 32x32 with 3x3 conv, pad 1, stride 1 (preserves size)
    layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
    ]
    sizes = compute_output_size(32, layers)
    assert sizes[-1][1] == 32, f"Test 1 FAILED: expected 32, got {sizes[-1][1]}"
    print("[OK] Test 1: 32x32 -> Conv3 pad1 stride1 -> 32x32")

    # Test 2: AlexNet first layer (from CS231n)
    # Input 227x227, filter 11x11, stride 4, no padding -> output 55x55
    layers = [
        {'type': 'conv', 'filter': 11, 'stride': 4, 'pad': 0, 'filters': 96},
    ]
    sizes = compute_output_size(227, layers)
    assert sizes[-1][1] == 55, f"Test 2 FAILED: expected 55, got {sizes[-1][1]}"
    print("[OK] Test 2: AlexNet first layer: 227 -> Conv11 stride4 -> 55")

    # Test 3: CONV + POOL
    layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'pool', 'filter': 2, 'stride': 2},
    ]
    sizes = compute_output_size(32, layers)
    assert sizes[-1][1] == 16, f"Test 3 FAILED: expected 16, got {sizes[-1][1]}"
    print("[OK] Test 3: 32 -> Conv3 pad1 -> 32 -> Pool2 stride2 -> 16")

    # Test 4: VGG-style block
    layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'pool', 'filter': 2, 'stride': 2},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 128},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 128},
        {'type': 'pool', 'filter': 2, 'stride': 2},
    ]
    sizes = compute_output_size(224, layers)
    assert sizes[-1][1] == 56, f"Test 4 FAILED: expected 56, got {sizes[-1][1]}"
    assert sizes[-1][2] == 128, f"Test 4 FAILED: expected depth 128, got {sizes[-1][2]}"
    print("[OK] Test 4: VGG block: 224 -> [Conv3 pad1]*2 -> Pool -> [Conv3 pad1]*2 -> Pool -> 56x56x128")

    # Test 5: Full VGG-16 spatial progression (from CS231n case study)
    vgg_layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'pool', 'filter': 2, 'stride': 2},  # 112
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 128},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 128},
        {'type': 'pool', 'filter': 2, 'stride': 2},  # 56
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 256},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 256},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 256},
        {'type': 'pool', 'filter': 2, 'stride': 2},  # 28
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'pool', 'filter': 2, 'stride': 2},  # 14
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'pool', 'filter': 2, 'stride': 2},  # 7
    ]
    sizes = compute_output_size(224, vgg_layers)
    assert sizes[-1][1] == 7, f"Test 5 FAILED: VGG final spatial size should be 7, got {sizes[-1][1]}"
    assert sizes[-1][2] == 512, f"Test 5 FAILED: VGG final depth should be 512, got {sizes[-1][2]}"
    print("[OK] Test 5: Full VGG-16: 224 -> ... -> 7x7x512")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
