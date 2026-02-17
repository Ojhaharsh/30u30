"""
solution_03_output_sizes.py - Solution for Exercise 3

Computes CNN output dimensions layer by layer.

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""


def compute_output_size(input_size, layers):
    """
    Compute spatial dimensions and depth at each layer.

    CONV: output = (W - F + 2P) / S + 1
    POOL: output = (W - F) / S + 1
    """
    sizes = [("Input", input_size, 3)]
    current_size = input_size
    current_depth = 3

    for i, layer in enumerate(layers):
        if layer['type'] == 'conv':
            new_size = (current_size - layer['filter'] + 2 * layer['pad']) / layer['stride'] + 1
            current_depth = layer['filters']
        elif layer['type'] == 'pool':
            new_size = (current_size - layer['filter']) / layer['stride'] + 1

        # Check for validity — fractional output means bad hyperparameters
        assert new_size == int(new_size), \
            f"Layer {i+1}: fractional output size {new_size} (invalid architecture)"

        current_size = int(new_size)
        sizes.append((f"Layer {i+1} ({layer['type']})", current_size, current_depth))

    return sizes


def check():
    """Test against CS231n examples."""

    # Test 1: 32x32 with 3x3 conv, pad 1, stride 1
    layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
    ]
    sizes = compute_output_size(32, layers)
    assert sizes[-1][1] == 32
    print("[OK] Test 1: 32x32 -> Conv3 pad1 stride1 -> 32x32")

    # Test 2: AlexNet first layer
    layers = [
        {'type': 'conv', 'filter': 11, 'stride': 4, 'pad': 0, 'filters': 96},
    ]
    sizes = compute_output_size(227, layers)
    assert sizes[-1][1] == 55
    print("[OK] Test 2: AlexNet: 227 -> Conv11 stride4 -> 55")

    # Test 3: CONV + POOL
    layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'pool', 'filter': 2, 'stride': 2},
    ]
    sizes = compute_output_size(32, layers)
    assert sizes[-1][1] == 16
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
    assert sizes[-1][1] == 56
    assert sizes[-1][2] == 128
    print("[OK] Test 4: VGG block: 224 -> ... -> 56x56x128")

    # Test 5: Full VGG-16
    vgg_layers = [
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 64},
        {'type': 'pool', 'filter': 2, 'stride': 2},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 128},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 128},
        {'type': 'pool', 'filter': 2, 'stride': 2},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 256},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 256},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 256},
        {'type': 'pool', 'filter': 2, 'stride': 2},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'pool', 'filter': 2, 'stride': 2},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'conv', 'filter': 3, 'stride': 1, 'pad': 1, 'filters': 512},
        {'type': 'pool', 'filter': 2, 'stride': 2},
    ]
    sizes = compute_output_size(224, vgg_layers)
    assert sizes[-1][1] == 7
    assert sizes[-1][2] == 512
    print("[OK] Test 5: Full VGG-16: 224 -> ... -> 7x7x512")

    print("\nAll tests passed.")


if __name__ == "__main__":
    check()
