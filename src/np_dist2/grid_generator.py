import numpy as np


def generate_fcc_lattice(size):
    """
    Generate a 3D FCC lattice of cubic shape containing complete elementary cells.

    Args:
        size (int): Number of unit cells along one edge of the cube

    Returns:
        numpy.ndarray: Array of shape (n_points, 3) containing all lattice points
    """
    # Basic FCC lattice points within one unit cell
    basis_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Corner point
            [0.5, 0.5, 0.0],  # Face center 1
            [0.5, 0.0, 0.5],  # Face center 2
            [0.0, 0.5, 0.5],  # Face center 3
        ]
    )

    # Create meshgrid for the cube
    x = np.arange(size)
    y = np.arange(size)
    z = np.arange(size)

    # Create all possible combinations of unit cell positions
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Reshape to get all positions
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Initialize array to store all points
    all_points = []

    # For each position in the cube
    for i in range(len(X)):
        # For each basis point in the unit cell
        for basis_point in basis_points:
            point = np.array(
                [X[i] + basis_point[0], Y[i] + basis_point[1], Z[i] + basis_point[2]]
            )
            all_points.append(point)

    return np.array(all_points)


# def generate_fcc_lattice(N):
#     """
#     Generate a 3D FCC lattice with a cubic shape.

#     Parameters:
#     N (int): Number of unit cells along each cube axis.

#     Returns:
#     np.array: A 2D numpy array where each row is a 3D point in the FCC lattice.
#     """
#     # Generate grid of unit cell indices (i, j, k)
#     i, j, k = np.mgrid[0:N, 0:N, 0:N]
#     cells = np.stack([i, j, k], axis=-1).reshape(-1, 3)

#     # FCC basis vectors within a unit cell
#     basis = np.array(
#         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
#     )

#     # Compute all points by adding basis vectors to each cell
#     points = (cells[:, np.newaxis, :] + basis).reshape(-1, 3)

#     return points


# def gen_fcc_grid(size):
#     grid = gen_unit_grid(size)
#     fc = gen_unit_grid(size)
#     sub1 = shift_grid(fc, [0.5, 0.5, 0])
#     sub2 = shift_grid(fc, [0.5, 0, 0.5])
#     sub3 = shift_grid(fc, [0, 0.5, 0.5])
#     fcc = np.vstack((grid, sub1, sub2, sub3))


def gen_unit_grid(size: int):
    return generate_3d_grid(size + 1, 0, size)


def generate_sphere_points(N: int):
    """Generate N points uniformly distributed on the surface of a unit sphere."""
    vec = np.random.randn(3, N)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def validate_grid_parameters(size: int, min_val: float, max_val: float) -> None:
    """Validate parameters for grid generation."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer")
    if min_val >= max_val:
        raise ValueError("Minimum value must be less than maximum value")


def shift_grid(grid, vec):
    vec = np.array(vec)
    return grid + np.ones(grid.shape) * vec


def grid_origin(grid):
    X, Y, Z = grid[:, 0], grid[:, 1], grid[:, 2]
    return np.array([np.mean(X), np.mean(Y), np.mean(Z)])


def shift_grid_to_origin(grid: np.ndarray) -> np.ndarray:
    origin = grid_origin(grid)
    return shift_grid(grid, -origin)


def generate_3d_grid(
    size: int, min_val: float = 0.0, max_val: float = 1.0
) -> np.ndarray:
    """
    Generate a 3D square grid where each row contains 3D vectors of grid points.

    Args:
        size: Number of points along each dimension
        min_val: Minimum coordinate value
        max_val: Maximum coordinate value

    Yields:
        Tuple containing (index, array of shape (size, 3) representing 3D coordinates)
    """
    validate_grid_parameters(size, min_val, max_val)

    # Calculate step size for uniform spacing
    step = (max_val - min_val) / (size - 1) if size > 1 else 0

    # Generate coordinates for each axis
    coords = np.arange(min_val, max_val + step, step)[:size]

    # Create meshgrid for all combinations
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Reshape into (size*size*size, 3) array of points
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

    return points
