# Test cases
import pytest
import numpy as np
from np_dist2.grid_generator import (
    generate_3d_grid,
    gen_unit_grid,
    shift_grid,
    grid_origin,
    shift_grid_to_origin,
)


def test_grid_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        generate_3d_grid(0)
    with pytest.raises(ValueError):
        generate_3d_grid(1, min_val=1.0, max_val=0.0)


def test_grid_shape():
    """Test grid shape and values."""
    size = 2
    min_val, max_val = 0.0, 1.0

    grid_gen = generate_3d_grid(size, min_val, max_val)

    points = grid_gen
    assert points.shape == (size**3, 3)

    # Verify coordinates are within bounds
    assert np.all(points >= min_val)
    assert np.all(points <= max_val)


def test_grid_points():
    """Test correct number of points generated."""
    size = 2
    points_list = generate_3d_grid(size)
    assert len(points_list) == size * size * size


def test_gen_unit():
    size = 3
    grid = gen_unit_grid(size)
    assert len(grid) == (size + 1) ** 3
    assert np.all(grid >= 0)
    assert np.all(grid <= size)
    assert np.max(grid) == size


unit_grid = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)

unit_grid_all = np.array(
    [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.5],
        [0.5, 1.5, 0.5],
        [0.5, 1.5, 1.5],
        [1.5, 0.5, 0.5],
        [1.5, 0.5, 1.5],
        [1.5, 1.5, 0.5],
        [1.5, 1.5, 1.5],
    ]
)

unit_grid3 = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.5],
        [0.0, 1.0, 0.5],
        [0.0, 1.0, 1.5],
        [1.0, 0.0, 0.5],
        [1.0, 0.0, 1.5],
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 1.5],
    ]
)

unit_grid2 = np.array(
    [
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 1.0],
        [0.0, 1.5, 0.0],
        [0.0, 1.5, 1.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 1.0],
        [1.0, 1.5, 0.0],
        [1.0, 1.5, 1.0],
    ]
)

unit_grid1 = np.array(
    [
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 1.0],
        [0.5, 1.0, 0.0],
        [0.5, 1.0, 1.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.0],
        [1.5, 1.0, 0.0],
        [1.5, 1.0, 1.0],
    ]
)


def test_sift_test():
    assert np.allclose(unit_grid1, shift_grid(unit_grid, np.array([0.5, 0, 0])))
    assert np.allclose(unit_grid2, shift_grid(unit_grid, np.array([0, 0.5, 0])))
    assert np.allclose(unit_grid3, shift_grid(unit_grid, np.array([0, 0, 0.5])))
    assert np.allclose(unit_grid_all, shift_grid(unit_grid, np.array([0.5, 0.5, 0.5])))


def test_grid_origin():
    grid = gen_unit_grid(1)
    origin = grid_origin(grid)
    assert np.allclose([0.5, 0.5, 0.5], origin)

    grid2 = shift_grid_to_origin(grid)
    assert np.allclose([0.0, 0.0, 0.0], grid_origin(grid2))


# Example usage
if __name__ == "__main__":
    size = 2
    grid_gen = generate_3d_grid(size, 0, size - 1)

    print(f"Generating {size}x{size}x{size} grid: \n{grid_gen}")
