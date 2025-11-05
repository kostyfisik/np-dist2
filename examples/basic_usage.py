"""Basic usage example of the np-dist2 library."""

import numpy as np
from np_dist2.grid_generator import gen_unit_grid
from np_dist2.lattice import get_lat_by_id
from np_dist2.rotate3d import rotate_vector


def main():
    """Demonstrate basic usage of the np-dist2 library."""
    print("Np Dist2 - Basic Usage Example")
    print("=" * 40)

    # Generate a simple 3D grid
    print("\n1. Generating a 3D grid...")
    grid = gen_unit_grid(2)
    print(f"Generated grid with {len(grid)} points")
    print(f"First 5 points: {grid[:5]}")

    # Analyze lattice properties
    print("\n2. Analyzing lattice properties...")
    r, lat, num = get_lat_by_id(0, grid)
    print(f"Lattice parameters for point 0:")
    print(f"  Distance from origin: {r}")
    print(f"  Average lattice spacing: {lat}")
    print(f"  Number of neighbors: {num}")

    # Demonstrate 3D rotation
    print("\n3. Demonstrating 3D rotation...")
    vector = np.array([1, 0, 0])
    rotated = rotate_vector(vector, np.pi/4, np.pi/4)
    print(f"Original vector: {vector}")
    print(f"Rotated vector: {rotated}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()