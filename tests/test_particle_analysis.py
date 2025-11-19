"""Tests for particle analysis functions."""

import numpy as np
import pytest
from np_dist2.analysis import (
    generate_random_directions,
    find_atoms_in_cylinder,
    calculate_effective_radius,
    find_near_neighbors,
    calculate_lattice_parameter_distribution,
    convert_lattice_to_density,
)


class TestGenerateRandomDirections:
    """Tests for generate_random_directions function."""

    def test_output_shape(self):
        """Verify output array has correct shape."""
        num_directions = 100
        directions = generate_random_directions(num_directions)
        assert directions.shape == (num_directions, 3)

    def test_unit_vectors(self):
        """Verify all generated vectors are unit vectors."""
        num_directions = 50
        directions = generate_random_directions(num_directions)
        norms = np.linalg.norm(directions, axis=1)
        assert np.allclose(norms, 1.0)

    def test_small_sample(self):
        """Test with a small number of directions."""
        directions = generate_random_directions(5)
        assert directions.shape == (5, 3)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


class TestFindAtomsInCylinder:
    """Tests for find_atoms_in_cylinder function."""

    def test_z_axis_cylinder(self):
        """Test cylinder along z-axis with known atom distribution."""
        # Create a 3x3x3 grid
        atoms = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    atoms.append([i * 2.0, j * 2.0, k * 2.0])
        atoms = np.array(atoms)

        # Cylinder along z-axis with radius 1.5
        axis = np.array([0.0, 0.0, 1.0])
        radius = 1.5

        cylinder_atoms = find_atoms_in_cylinder(atoms, axis, radius)

        # Should include atoms with x^2 + y^2 <= radius^2
        # Center column: (0,0,k) - 3 atoms
        # Expected: only atoms where sqrt(x^2 + y^2) <= 1.5
        assert len(cylinder_atoms) > 0

        # Verify all selected atoms satisfy the cylinder condition
        projections = np.dot(cylinder_atoms, axis)[:, np.newaxis] * axis
        perpendicular = cylinder_atoms - projections
        perp_distances = np.linalg.norm(perpendicular, axis=1)
        assert np.all(perp_distances <= radius + 1e-10)

    def test_diagonal_axis_cylinder(self):
        """Test cylinder along diagonal axis."""
        # Create atoms along and near the [1,1,0] direction
        atoms = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],  # Perpendicular offset
            [0.0, 1.0, 0.0],  # Perpendicular offset
        ])

        axis = np.array([1.0, 1.0, 0.0])
        radius = 0.5

        cylinder_atoms = find_atoms_in_cylinder(atoms, axis, radius)

        # First three atoms are along the axis, should be included
        # Last two are at perpendicular distance ~0.707, should be excluded
        assert len(cylinder_atoms) >= 3

        # Verify cylinder condition
        axis_normalized = axis / np.linalg.norm(axis)
        projections = np.dot(cylinder_atoms, axis_normalized)[:, np.newaxis] * axis_normalized
        perpendicular = cylinder_atoms - projections
        perp_distances = np.linalg.norm(perpendicular, axis=1)
        assert np.all(perp_distances <= radius + 1e-10)

    def test_empty_cylinder(self):
        """Test with no atoms inside the cylinder."""
        atoms = np.array([
            [10.0, 10.0, 0.0],
            [10.0, -10.0, 0.0],
            [-10.0, 10.0, 0.0],
            [-10.0, -10.0, 0.0],
        ])

        axis = np.array([0.0, 0.0, 1.0])
        radius = 0.5

        cylinder_atoms = find_atoms_in_cylinder(atoms, axis, radius)

        assert len(cylinder_atoms) == 0
        assert cylinder_atoms.shape == (0, 3)

    def test_empty_input(self):
        """Test with empty atom array."""
        atoms = np.array([]).reshape(0, 3)
        axis = np.array([1.0, 0.0, 0.0])
        radius = 1.0

        result = find_atoms_in_cylinder(atoms, axis, radius)

        assert result.shape == (0, 3)


class TestCalculateEffectiveRadius:
    """Tests for calculate_effective_radius function."""

    def test_perfect_sphere(self):
        """Test with atoms arranged in a perfect sphere."""
        # Create atoms distributed on a sphere of radius R
        R = 10.0
        num_atoms = 200

        # Generate random points on sphere surface
        phi = np.random.uniform(0, 2 * np.pi, num_atoms)
        theta = np.arccos(np.random.uniform(-1, 1, num_atoms))

        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)

        atoms = np.column_stack([x, y, z])

        eff_radius, all_radii = calculate_effective_radius(
            atoms, num_directions=50, cylinder_radius=R * 0.3
        )

        # Effective radius should be close to R
        assert np.isclose(eff_radius, R, rtol=0.15)
        assert len(all_radii) == 50

    def test_empty_atoms(self):
        """Test with empty atom array."""
        atoms = np.array([]).reshape(0, 3)

        eff_radius, all_radii = calculate_effective_radius(
            atoms, num_directions=10, cylinder_radius=1.0
        )

        assert eff_radius == 0.0
        assert len(all_radii) == 0

    def test_single_atom(self):
        """Test with a single atom at origin."""
        atoms = np.array([[0.0, 0.0, 0.0]])

        eff_radius, all_radii = calculate_effective_radius(
            atoms, num_directions=10, cylinder_radius=1.0
        )

        # All radii should be 0 (atom at origin)
        assert eff_radius == 0.0
        assert np.all(all_radii == 0.0)

    def test_return_types(self):
        """Verify correct return types."""
        atoms = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        eff_radius, all_radii = calculate_effective_radius(
            atoms, num_directions=20, cylinder_radius=0.5
        )

        assert isinstance(eff_radius, (float, np.floating))
        assert isinstance(all_radii, np.ndarray)
        assert len(all_radii) == 20


class TestFindNearNeighbors:
    """Tests for find_near_neighbors function."""

    def test_fcc_interior_atom(self, perfect_fcc_lattice):
        """Test with an atom deep inside a perfect FCC lattice."""
        atoms = perfect_fcc_lattice

        # Find an interior atom (not on edges)
        # Use an atom that should have 12 nearest neighbors
        # Let's use an atom that's definitely interior
        center_idx = len(atoms) // 2

        neighbor_indices, d_min = find_near_neighbors(center_idx, atoms, dist_factor=1.2)

        # In a perfect FCC, interior atoms have 12 nearest neighbors
        # With dist_factor=1.2, we should get exactly these 12
        assert d_min > 0
        assert len(neighbor_indices) >= 4  # At minimum should have some neighbors
        assert len(neighbor_indices) <= 12  # Should not exceed FCC coordination number

    def test_surface_atom(self):
        """Test with an atom on the surface with fewer neighbors."""
        # Create a small FCC-like cluster
        atoms = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])

        neighbor_indices, d_min = find_near_neighbors(0, atoms, dist_factor=1.2)

        # First atom has 3 nearest neighbors at distance 1.0
        assert np.isclose(d_min, 1.0, rtol=0.01)
        assert len(neighbor_indices) >= 3

    def test_isolated_atom(self):
        """Test with a single atom (no neighbors)."""
        atoms = np.array([[0.0, 0.0, 0.0]])

        neighbor_indices, d_min = find_near_neighbors(0, atoms, dist_factor=1.2)

        assert len(neighbor_indices) == 0
        assert d_min == 0.0

    def test_two_atoms(self):
        """Test with exactly two atoms."""
        atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        neighbor_indices, d_min = find_near_neighbors(0, atoms, dist_factor=1.5)

        assert len(neighbor_indices) == 1
        assert neighbor_indices[0] == 1
        assert np.isclose(d_min, 1.0)

    def test_dist_factor_effect(self):
        """Test that dist_factor properly controls neighbor selection."""
        atoms = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Distance 1.0
            [2.0, 0.0, 0.0],  # Distance 2.0
            [0.0, 1.0, 0.0],  # Distance 1.0
        ])

        # With dist_factor=1.1, should only get atoms at distance ~1.0
        neighbors_narrow, d_min = find_near_neighbors(0, atoms, dist_factor=1.1)
        assert len(neighbors_narrow) == 2

        # With dist_factor=2.5, should get all three neighbors
        neighbors_wide, d_min = find_near_neighbors(0, atoms, dist_factor=2.5)
        assert len(neighbors_wide) == 3

    def test_invalid_index(self):
        """Test with invalid atom index."""
        atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        with pytest.raises(ValueError):
            find_near_neighbors(5, atoms)

        with pytest.raises(ValueError):
            find_near_neighbors(-1, atoms)


class TestCalculateLatticeParameterDistribution:
    """Tests for calculate_lattice_parameter_distribution function."""

    def test_perfect_fcc_lattice(self, perfect_fcc_lattice):
        """Test with perfect FCC lattice."""
        atoms = perfect_fcc_lattice
        lattice_constant = 4.0896

        # Expected nearest neighbor distance for FCC: a * sqrt(2) / 2
        expected_nn_distance = lattice_constant * np.sqrt(2) / 2

        all_r, all_a_local = calculate_lattice_parameter_distribution(
            atoms, num_directions=30, cylinder_radius=5.0
        )

        # Should have collected data for multiple atoms
        assert len(all_r) > 0
        assert len(all_a_local) > 0
        assert len(all_r) == len(all_a_local)

        # For interior atoms, local lattice parameter should be close to expected
        # Filter for interior atoms (r < max(r) * 0.7)
        max_r = np.max(all_r)
        interior_mask = all_r < max_r * 0.7

        if np.any(interior_mask):
            interior_a_local = all_a_local[interior_mask]
            mean_a_local = np.mean(interior_a_local)
            # Should be close to theoretical value
            assert np.isclose(mean_a_local, expected_nn_distance, rtol=0.15)

    def test_equal_length_arrays(self):
        """Verify that returned arrays have equal length."""
        # Create a simple cubic arrangement
        atoms = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        all_r, all_a_local = calculate_lattice_parameter_distribution(
            atoms, num_directions=20, cylinder_radius=1.0
        )

        assert len(all_r) == len(all_a_local)

    def test_empty_atoms(self):
        """Test with empty atom array."""
        atoms = np.array([]).reshape(0, 3)

        all_r, all_a_local = calculate_lattice_parameter_distribution(
            atoms, num_directions=10, cylinder_radius=1.0
        )

        assert len(all_r) == 0
        assert len(all_a_local) == 0

    def test_single_direction(self):
        """Test with a single direction."""
        atoms = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])

        all_r, all_a_local = calculate_lattice_parameter_distribution(
            atoms, num_directions=1, cylinder_radius=0.5
        )

        # Should process at least some atoms
        assert isinstance(all_r, np.ndarray)
        assert isinstance(all_a_local, np.ndarray)


class TestConvertLatticeToDensity:
    """Tests for convert_lattice_to_density function."""

    def test_silver_fcc_density(self):
        """Test with known Silver FCC lattice constant."""
        # Silver lattice constant: a = 4.08 Angstroms
        # Nearest neighbor distance: d = a * sqrt(2) / 2
        a_silver = 4.08
        d = a_silver * np.sqrt(2) / 2

        lattice_params = np.array([d])
        density = convert_lattice_to_density(lattice_params, structure='fcc')

        # Theoretical density: 4 / a^3
        expected_density = 4.0 / (a_silver ** 3)

        assert np.isclose(density[0], expected_density, rtol=0.01)

    def test_array_processing(self):
        """Test that the function processes arrays correctly."""
        # Create an array of lattice parameters
        lattice_params = np.array([2.0, 2.5, 3.0, 3.5])

        densities = convert_lattice_to_density(lattice_params, structure='fcc')

        # Verify output is an array of the same length
        assert len(densities) == len(lattice_params)
        assert isinstance(densities, np.ndarray)

        # Verify densities decrease as lattice parameter increases
        assert np.all(densities[:-1] >= densities[1:])

    def test_mathematical_relationship(self):
        """Verify the mathematical relationship d = a*sqrt(2)/2 and n = 4/a^3."""
        d = 3.0  # Arbitrary nearest neighbor distance

        lattice_params = np.array([d])
        density = convert_lattice_to_density(lattice_params, structure='fcc')

        # Calculate expected values
        a = d * np.sqrt(2)
        expected_density = 4.0 / (a ** 3)

        assert np.isclose(density[0], expected_density, rtol=1e-10)

    def test_unsupported_structure(self):
        """Test that unsupported structure types raise an error."""
        lattice_params = np.array([2.0, 3.0])

        with pytest.raises(ValueError, match="Unsupported structure type"):
            convert_lattice_to_density(lattice_params, structure='bcc')

        with pytest.raises(ValueError, match="Unsupported structure type"):
            convert_lattice_to_density(lattice_params, structure='hcp')

    def test_single_value(self):
        """Test with a single lattice parameter."""
        d = 2.5
        result = convert_lattice_to_density(np.array([d]), structure='fcc')

        assert len(result) == 1
        assert result[0] > 0
