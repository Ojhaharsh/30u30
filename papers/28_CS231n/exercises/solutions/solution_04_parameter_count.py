"""
solution_04_parameter_count.py - Solution for Exercise 4

Counts CNN parameters with VGGNet-16 breakdown.

Reference: CS231n - https://cs231n.github.io/convolutional-networks/
"""


def count_parameters(architecture):
    """
    Count learnable parameters for each layer.

    CONV: (F * F * D_in + 1) * K  (weights + bias per filter)
    POOL: 0
    FC:   (D_in + 1) * D_out     (weights + bias per output)
    """
    results = []
    total = 0

    for layer in architecture:
        if layer['type'] == 'conv':
            f = layer['filter_size']
            d_in = layer['in_depth']
            k = layer['filters']
            params = (f * f * d_in + 1) * k
            results.append((f"CONV{f}-{k}", params))
            total += params
        elif layer['type'] == 'pool':
            results.append(("POOL", 0))
        elif layer['type'] == 'fc':
            d_in = layer['in_size']
            d_out = layer['out_size']
            params = (d_in + 1) * d_out
            results.append((f"FC-{d_out}", params))
            total += params

    return results, total


def check():
    """Test against CS231n's VGGNet-16 parameter counts."""

    # Test 1: Single CONV
    arch = [{'type': 'conv', 'in_depth': 3, 'filters': 64, 'filter_size': 3}]
    results, total = count_parameters(arch)
    assert total == 1792
    print("[OK] Test 1: Conv3-64 with RGB input = 1,792 params")

    # Test 2: FC layer
    arch = [{'type': 'fc', 'in_size': 4096, 'out_size': 4096}]
    results, total = count_parameters(arch)
    assert total == (4096 + 1) * 4096
    print("[OK] Test 2: FC 4096->4096 = 16,781,312 params")

    # Test 3: Pool
    arch = [{'type': 'pool'}]
    results, total = count_parameters(arch)
    assert total == 0
    print("[OK] Test 3: Pool = 0 params")

    # Test 4: VGG first CONV
    arch = [{'type': 'conv', 'in_depth': 3, 'filters': 64, 'filter_size': 3}]
    results, total = count_parameters(arch)
    print(f"[OK] Test 4: VGG first CONV: {total} params")

    # Test 5: VGG first FC
    arch = [{'type': 'fc', 'in_size': 7*7*512, 'out_size': 4096}]
    results, total = count_parameters(arch)
    expected = (7*7*512 + 1) * 4096
    assert total == expected
    print(f"[OK] Test 5: VGG first FC = {total:,} params")

    # Test 6: Full VGG-16
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
