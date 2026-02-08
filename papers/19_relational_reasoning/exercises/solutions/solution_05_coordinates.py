import torch
from implementation import add_coordinates


def test_spatial_logic():
    """
    Solution 5: Verify the coordinate injection utility expected by
    the exercise `test_spatial_logic()`.

    Checks:
    - Returned shape is (B, N, D+2)
    - First object's x-coordinate equals -1.0 (normalized grid)
    """
    batch, N, D = 1, 4, 8
    objs = torch.randn(batch, N, D)

    objs_with_coords = add_coordinates(objs)

    # Dimension check
    assert objs_with_coords.shape == (batch, N, D + 2), (
        f"Expected shape {(batch, N, D+2)}, got {objs_with_coords.shape}"
    )

    # x-coordinates are created as linspace(-1, 1, N). First should be -1.0
    first_x = objs_with_coords[0, 0, -2].item()
    assert abs(first_x + 1.0) < 1e-6, f"Expected first x coord ~ -1.0, got {first_x}"

    print("Solution 5: [PASS]")
