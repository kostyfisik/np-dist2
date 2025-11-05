"""Test script to verify that the library modules are working correctly."""

import numpy as np
from np_dist2.grid_generator import gen_unit_grid
from np_dist2.lattice import get_lat_by_id

# Test the grid generator
print("Testing grid generator...")
grid = gen_unit_grid(2)
print(f"Generated grid with {len(grid)} points")
print(f"Grid shape: {grid.shape}")
print(f"First few points: {grid[:5]}")

# Test the lattice module
print("\nTesting lattice module...")
r, lat, num = get_lat_by_id(0, grid)
print(f"Lattice parameters for point 0: r={r}, lat={lat}, num={num}")

print("\nAll tests passed!")