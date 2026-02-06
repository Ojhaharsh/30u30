"""
Solution 1: KC Approximation via gzip
"""

import numpy as np
import gzip


def gzip_complexity(data: bytes, level: int = 9) -> int:
    """Approximate KC using gzip compressed size."""
    return len(gzip.compress(data, compresslevel=level))


def test_known_signals():
    """Test on three signals with known properties."""
    # All zeros -- highly compressible
    zeros = bytes(1024)
    kc_zeros = gzip_complexity(zeros)
    print(f"  All zeros (1024 bytes): {kc_zeros} compressed bytes")

    # Random bytes -- incompressible
    random_data = np.random.randint(0, 256, size=1024, dtype=np.uint8).tobytes()
    kc_random = gzip_complexity(random_data)
    print(f"  Random    (1024 bytes): {kc_random} compressed bytes")

    # Repeating pattern -- intermediate
    pattern = (b"ABCD" * 256)
    kc_pattern = gzip_complexity(pattern)
    print(f"  Pattern   (1024 bytes): {kc_pattern} compressed bytes")

    # Verify ordering
    assert kc_zeros < kc_pattern < kc_random, (
        f"Expected zeros < pattern < random, "
        f"got {kc_zeros} < {kc_pattern} < {kc_random}"
    )
    print("  [OK] Ordering verified: zeros < pattern < random")


def compare_compression_levels():
    """Compare gzip levels on the same data."""
    data = (b"Hello World! " * 340)[:4096]  # 4096 bytes

    print(f"  Data size: {len(data)} bytes")
    prev_size = float('inf')
    for level in [1, 3, 5, 7, 9]:
        size = gzip_complexity(data, level=level)
        print(f"  Level {level}: {size} bytes")
        assert size <= prev_size + 5, (
            f"Level {level} should not be much larger than level {level-2}"
        )
        prev_size = size

    print("  [OK] Higher levels give tighter bounds (as expected)")


if __name__ == "__main__":
    print("Solution 1: KC Approximation via gzip")
    print("=" * 40)
    print()

    print("Task 1: Testing known signals...")
    test_known_signals()
    print()

    print("Task 2: Comparing compression levels...")
    compare_compression_levels()
